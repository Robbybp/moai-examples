import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence as MPIN
import SparseArrays
import Profile

import Printf
import DataFrames
import CSV

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("models.jl")

function solve_with_ma57(csc::SparseArrays.SparseMatrixCSC, rhs::Vector)
    solver = MadNLPHSL.Ma57Solver(csc)
    _t = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - _t
    sol = copy(rhs)
    _t = time()
    MadNLP.solve!(solver, sol)
    t_solve = time() - _t
    res = (;
        time = (;
            factorize = t_factorize,
            solve = t_solve,
        ),
        solution = sol,
    )
    return res
end

function fill_upper_triangle(csc::SparseArrays.SparseMatrixCSC)
    I, J, V = SparseArrays.findnz(csc)
    strict_lower = filter(k -> I[k] > J[k], 1:length(I))
    upper_I = J[strict_lower]
    upper_J = I[strict_lower]
    upper_V = V[strict_lower]
    append!(I, upper_I)
    append!(J, upper_J)
    append!(V, upper_V)
    new = SparseArrays.sparse(I, J, V)
    return new
end

IMAGE_INDEX = 7
ADVERSARIAL_LABEL = 1
THRESHOLD = 0.6

nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu1024nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu2048nodes4layers.pt")
model, outputs, formulation = get_adversarial_model(
    nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
    reduced_space = false
)
nlp, kkt_system, kkt_matrix = get_kkt(model, Solver=MadNLPHSL.Ma57Solver)

pivot_vars, pivot_cons = get_vars_cons(formulation)
pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
pivot_index_set = Set(pivot_indices)
@assert kkt_matrix.m == kkt_matrix.n
reduced_indices = filter(i -> !(i in pivot_index_set), 1:kkt_matrix.m)
pivot_dim = length(pivot_indices)
@assert pivot_dim % 2 == 0

P = pivot_indices
R = reduced_indices
C_orig = kkt_matrix[P, P]

display(C_orig)

# Filter out constraint regularization nonzeros
# By convention, constraints are the second half of the pivot indices
to_ignore = Set((pivot_dim / 2 + 1):pivot_dim)
I, J, V = SparseArrays.findnz(C_orig)
to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
I = I[to_retain]
J = J[to_retain]
V = V[to_retain]
C = SparseArrays.sparse(I, J, V, C_orig.m, C_orig.n)

println("Decomposable pivot matrix:")
display(C)

CHECK_MA57 = false
if CHECK_MA57
    res_reg = solve_with_ma57(C_orig, ones(C_orig.m))
    res_noreg = solve_with_ma57(C, ones(C_orig.m))

    println("With reg.,    NNZ = $(SparseArrays.nnz(C_orig))")
    println("Without reg., NNZ = $(SparseArrays.nnz(C))")

    println("Factorize/solve times")
    println("---------------------")
    println("With regularization nonzeros:")
    display(res_reg.time)
    println("Without regularization nonzeros:")
    display(res_noreg.time)
end

rhs = ones(C_orig.m, 3)
solver = MadNLPHSL.Ma57Solver(C_orig)
MadNLP.factorize!(solver)
sol = copy(rhs)
MadNLP.solve!(solver, sol)
display(sol)

#C = fill_upper_triangle(C)
#println("Full symmetric pivot matrix:")
#display(C)
#
#igraph = MPIN.IncidenceGraphInterface(C)
#blocks = MPIN.block_triangularize(igraph)
#row_order = vcat([cb for (cb, vb) in blocks]...)
#col_order = vcat([vb for (cb, vb) in blocks]...)
#C_bt = C[row_order, col_order]
#println("Pivot matrix in BT form")
#display(C_bt)
