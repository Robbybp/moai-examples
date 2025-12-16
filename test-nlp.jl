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
using Profile, PProf
import Serialization
import LinearAlgebra

include("linalg.jl")
include("btsolver.jl")
include("nlpmodels.jl")
include("btsolver.jl")
include("kkt-partition.jl")
include("model-getter.jl")

LinearAlgebra.BLAS.set_num_threads(1)
LinearAlgebra.BLAS.lbt_set_num_threads(1)

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

#model_name = "mnist"
#model_name = "scopf"
model_name = "lsv"

#nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu512nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-tanh512nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu600nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu768nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-tanh1024nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu1024nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-tanh2048nodes4layers.pt")

#nnfile = joinpath("nn-models", "scopf", "100nodes3layers.pt")
#nnfile = joinpath("nn-models", "scopf", "500nodes5layers.pt")
#nnfile = joinpath("nn-models", "scopf", "1000nodes7layers.pt")
#nnfile = joinpath("nn-models", "scopf", "relu100nodes3layers.pt")
#nnfile = joinpath("nn-models", "scopf", "relu500nodes5layers.pt")
#nnfile = joinpath("nn-models", "scopf", "relu1000nodes7layers.pt")
#nnfile = joinpath("nn-models", "scopf", "relu2000nodes20layers.pt")

nnfile = joinpath("nn-models", "lsv", "118_bus", "118_bus_2048node.pt")

_t = time()
m, formulation = MODEL_GETTER[model_name](nnfile)
dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Make model")

variables, constraints = get_vars_cons(formulation)
pivot_indices = convert(Vector{Int32}, get_kkt_indices(m, variables, constraints))
blocks = partition_indices_by_layer(m, formulation; indices = pivot_indices)
dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Extract indices")

println("Solving with Schur+BT")
madnlp_schur = JuMP.optimizer_with_attributes(
    MadNLP.Optimizer,
    "tol" => 1e-6,
    #"linear_solver" => MadNLPHSL.Ma57Solver,
    "linear_solver" => SchurComplementSolver,
    "pivot_indices" => pivot_indices,
    "PivotSolver" => BlockTriangularSolver,
    "pivot_solver_opt" => BlockTriangularOptions(; blocks),
    "ReducedSolver" => MadNLPHSL.Ma57Solver,
    #"richardson_tol" => 1.0,
    #"richardson_acceptable_tol" => 1.0,
    #"inertia_correction_method" => MadNLP.InertiaIgnore,
    #"disable_garbage_collector" => true,

    "max_iter" => 10,
    "print_level" => MadNLP.INFO,
)
JuMP.set_optimizer(m, madnlp_schur)

PROFILE_ALLOCS = false
if PROFILE_ALLOCS
    Profile.Allocs.@profile sample_rate=0.0001 JuMP.optimize!(m)
    #Profile.print()
    #Profile.print(format = :flat, sortedby = :count, mincount = 100)
    PProf.Allocs.pprof()
end

PROFILE_RUNTIME = false
if PROFILE_RUNTIME
    Profile.@profile JuMP.optimize!(m)
    data, lidict = Profile.retrieve()
    # default is 62261
    PProf.pprof(data, lidict; webport = 62262)
    open("profdata.bin", "w") do io
        Serialization.serialize(io, (data, lidict))
    end
end

@timev JuMP.optimize!(m)

println(m.moi_backend.optimizer.model.solver.kkt.linear_solver.timer)

if true
    println()
    solvername = "MA86"
    solvertype = MadNLPHSL.Ma86Solver
    println("Solving with $solvername")
    madnlp = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "tol" => 1e-6,
        "linear_solver" => solvertype,
        "max_iter" => 10,
    )
    JuMP.set_optimizer(m, madnlp)
    @time JuMP.optimize!(m)
end
