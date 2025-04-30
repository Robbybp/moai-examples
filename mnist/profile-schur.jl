import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence
import SparseArrays
import Profile

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("models.jl")

PRECOMPILE = true
if PRECOMPILE
    model, info = make_tiny_model()
    _, _, kkt_matrix = get_kkt(model)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; pivot_indices)
    solver = SchurComplementSolver(kkt_matrix; opt)
    rhs = ones(kkt_matrix.m)
    MadNLP.factorize!(solver)
    MadNLP.solve!(solver, rhs)
end

nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
image_index = 7
adversarial_label = 1
threshold = 0.6

model, outputs, formulation = get_adversarial_model(
    nnfile, image_index, adversarial_label, threshold
)

nlp, kkt_system, kkt_matrix = get_kkt(model)
pivot_vars, pivot_cons = get_vars_cons(formulation)
pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
pivot_indices = convert(Vector{Int32}, pivot_indices)
rhs = ones(kkt_matrix.m)
opt = SchurComplementOptions(; pivot_indices)
solver = SchurComplementSolver(kkt_matrix; opt)

NSAMPLES = 2
for i in 1:NSAMPLES
    d = copy(rhs)
    # If we update values in the KKT system, we need to run the following:
    #update_kkt!(kkt_system, nlp; x)
    #MadNLP.build_kkt!(kkt_system)
    MadNLP.factorize!(solver)
    MadNLP.solve!(solver, d)
end
println(solver.timer)

t_init_start = time()
ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
t_ma27_init = time() - t_init_start

t_factorize_start = time()
MadNLP.factorize!(ma27)
t_ma27_factorize = time() - t_factorize_start

t_solve_start = time()
MadNLP.solve!(ma27, rhs)
t_ma27_solve = time() - t_solve_start

println("MA27 timing")
println("-----------")
println("initialize: $t_ma27_init")
println("factorize:  $t_ma27_factorize")
println("solve:      $t_ma27_solve")
