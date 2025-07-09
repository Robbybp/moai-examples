import MathOptAI as MOAI
import JuMP
import Ipopt
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays
using Test
using Printf

include("linalg.jl")
include("btsolver.jl")
include("nlpmodels.jl")
include("btsolver.jl")
include("kkt-partition.jl")
include("adversarial-image.jl")

function make_square_model(nnfile)
    datadir = joinpath("data", "MNIST", "raw")
    test_data = MLDatasets.MNIST(; split = :test, dir = datadir)

    length_dim, height_dim = size(test_data[1].features)
    xref = test_data[image_index].features
    target_label = test_data[image_index].targets

    println("LABEL FOR IMAGE $(image_index): $(target_label)")
    println("LOADING NN FROM FILE: $nnfile")
    torch = PythonCall.pyimport("torch")
    nn = torch.load(nnfile, weights_only = false)
    pyinput = torch.tensor(vec(xref))
    pyoutput = nn(pyinput).detach().numpy()
    output = PythonCall.pyconvert(Array, pyoutput)
    # Indices of `output` are 1-indexed, but predictions are 0-indexed
    prediction = argmax(output) - 1
    println("NN PREDICTS: $output")

    m = JuMP.Model()
    #JuMP.@variable(m, 0.0 <= x[1:height_dim, 1:length_dim] <= 1.0, start = 0.5)
    JuMP.@variable(m, 0.0 <= x[i in 1:height_dim, j in 1:length_dim] <= 1.0, start = xref[i, j])
    predictor = MOAI.PytorchModel(nnfile)
    config = Dict(:ReLU => MOAI.ReLUQuadratic(relaxation_parameter = 1e-6))
    y, formulation = MOAI.add_predictor(
        m,
        predictor,
        vec(x);
        config,
        hessian = true,
    )
    JuMP.@constraint(m, 0.0 .<= y .<= 1.0)
    JuMP.@constraint(m, x .== xref)
    # TODO: We could minimize outputs?
    JuMP.@objective(m, Min, 0.0)
    return m, y, formulation
end

nnfile = joinpath("nn-models", "mnist-relu2048nodes4layers.pt")
image_index = 7
adversarial_label = 1
threshold = 0.6
_t = time()
m, y, formulation = make_square_model(nnfile)
#m, y, formulation = get_adversarial_model(
#    nnfile,
#    image_index,
#    adversarial_label,
#    threshold,
#)
dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Make model")

variables, constraints = get_vars_cons(formulation)
pivot_indices = convert(Vector{Int32}, get_kkt_indices(m, variables, constraints))
blocks = partition_indices_by_layer(m, formulation; indices = pivot_indices)
dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Extract indices")

println("Solving with Schur+BT")
madnlp_schur = JuMP.optimizer_with_attributes(
    MadNLP.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => SchurComplementSolver,
    "pivot_indices" => pivot_indices,
    "PivotSolver" => BlockTriangularSolver,
    "pivot_solver_opt" => BlockTriangularOptions(; blocks),
    "ReducedSolver" => MadNLPHSL.Ma57Solver,
    #"disable_garbage_collector" => true,
    "max_iter" => 10,
)
JuMP.set_optimizer(m, madnlp_schur)
@time JuMP.optimize!(m)

println()
println("Solving with MA57")
madnlp_ma57 = JuMP.optimizer_with_attributes(
    MadNLP.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => MadNLPHSL.Ma57Solver,
    "max_iter" => 10,
)
JuMP.set_optimizer(m, madnlp_ma57)
@time JuMP.optimize!(m)
