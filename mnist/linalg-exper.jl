import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence
import SparseArrays

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")

nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
image_index = 7
adversarial_label = 1
threshold = 0.6

m, outputs, formulation = get_adversarial_model(
    nnfile, image_index, adversarial_label, threshold
)

# My goal is to get the KKT matrix as SparseMatrixCSC
# Then I will:
# - Factorize it with MA27, backsolve, and compute inertia
# - Identify coordinates corresponding to NN output and internal variables
# - Perform a Schur complement decomposition WRT these coordinates, factorize,
#   backsolve, and solve the Schur system
# - Compare solutions and make sure they are the same

# Construct NLPModel from JuMP model
nlp = NLPModelsJuMP.MathOptNLPModel(m)

# Construct MadNLP's callback API from NLPModels' callback API
cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)

# These are the indices of equality and inequality constraints/variables,
# but I have no way to know exactly which constraints these are.
ind_cons = MadNLP.get_index_constraints(nlp)

linear_solver = MadNLPHSL.Ma27Solver

# MadNLP creates a KKT *system* object rather than just returning a KKT *matrix*
# I will re-implement this method/struct to contain additional coordinates to
# reduce.
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    ind_cons,
    linear_solver,
)

# Initialize KKT system with "default values". I assume these are primal variable
# start values and... some default values for duals...
MadNLP.initialize!(kkt)

# Don't need to call build_kkt! when we just initialized?
#MadNLP.build_kkt!(kkt)
#
# This returns a *reference* to the internal KKT matrix. That is, values will be
# updated automatically.
#
# The KKT matrix is ordered into Hessian and Jacobian blocks, but the Jacobian
# block is not ordered by (eq, ineq). It is ordered in whatever order the constraints
# appear in the NLPModel.
kkt_matrix = MadNLP.get_kkt(kkt)

# Factors will be stored inside the linear_solver instance.
MadNLP.factorize!(kkt.linear_solver)

@assert MadNLP.is_inertia(kkt.linear_solver)
# Inertia is in the format (pos, neg, zero)
inertia = MadNLP.inertia(kkt.linear_solver)

# Now I need to backsolve and get a solution that I can check later
b = ones(size(kkt_matrix)[1])
x = copy(b)
MadNLP.solve!(kkt.linear_solver, x)
x_ma27 = x

# This is basically the workflow I need to go through.
# But how does this actually happen inside of MadNLP?
# - iterative refinement?
# - ReducedKKTSystem?
# Now I need to:
# - Implement this workflow with a Schur complement decomposition.
# - Figure out what methods I need to call to actually use a custom linear
#   solver during the solve.

vars, cons = get_vars_cons(formulation)
pivot_indices = get_kkt_indices(m, vars, cons)
pivot_indices = convert(Vector{Int32}, pivot_indices)
# TODO: Map these vars/cons to indices, apply offset, and provide these
# indices to SchurComplementOptions.

# linear_solver is constructed by KKTSystem?
# presumably I can use default_options? Or just pass this info in as
# options?

# ***THESE INDICES NEED TO BE THE SAME TYPE AS THE KKT_MATRIX INDEX TYPE***
#
# Now I need to get these indices from my JuMP model.
# - I will first need to know what coordinates they are in the NLPModel's Jacobian
# - Then I will need to know what coordinates they are in the KKT matrix
opt = SchurComplementOptions(; pivot_indices)
# This will be constructed as:
linear_solver = SchurComplementSolver(
    kkt_matrix;
    opt = opt,
    logger = MadNLP.MadNLPLogger(),
)

b = ones(size(kkt_matrix)[1])
x = copy(b)
println("Running with $(MadNLP.introduce(linear_solver))")
MadNLP.factorize!(linear_solver)
MadNLP.solve!(linear_solver, x)

d_diff = x - x_ma27
maxdiff = maximum(abs.(d_diff))
println("||ϵ|| = $maxdiff")
