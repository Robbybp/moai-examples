import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence as MPIN
import SparseArrays
import Profile

import SparseArrays: UMFPACK

import Printf
import DataFrames
import CSV

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("models.jl")
include("ma48.jl")

# TODO: Accept a matrix as the RHS
function solve_with_ma57(csc::SparseArrays.SparseMatrixCSC, rhs::Matrix)
    solver = MadNLPHSL.Ma57Solver(csc)
    _t = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - _t
    println("INFO after factorization:")
    display(collect(enumerate(solver.info)))
    sol = copy(rhs)
    _t = time()
    println("INFO after backsolve:")
    display(collect(enumerate(solver.info)))
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

function solve_with_umfpack(
    csc::SparseArrays.SparseMatrixCSC,
    rhs::Matrix;
    order::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}} = nothing,
)
    if order === nothing
        _t = time()
        igraph = MPIN.IncidenceGraphInterface(csc)
        blocks = MPIN.block_triangularize(igraph)
        row_order = vcat([cb for (cb, vb) in blocks]...)
        col_order = vcat([vb for (cb, vb) in blocks]...)
        t_symbolic = time() - _t
    else
        row_order, col_order = order
        t_symbolic = 0.0
    end
    csc = csc[row_order, col_order]
    REAL = eltype(csc.nzval)
    INT = eltype(csc.rowval)
    umfpack_control = UMFPACK.get_umfpack_control(REAL, INT)
    # Set ordering to None.
    # From user guide: https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/UMFPACK/Doc/UMFPACK_QuickStart.pdf
    umfpack_control[UMFPACK.JL_UMFPACK_ORDERING] = 5.0
    # Unsymmetric strategy.
    # (I know this by trial and error with UMFPACK.show_umf_ctrl)
    umfpack_control[6] = 1.0
    _t = time()
    lu = LinearAlgebra.lu(csc; control = umfpack_control)
    t_factorize = time() - _t
    # TODO: Performance gain with `rdiv!`?
    # (which will require copying RHS)
    _t = time()
    solution = lu \ rhs
    t_solve = time() - _t
    res = (;
        time = (;
            symbolic = t_symbolic,
            factorize = t_factorize,
            solve = t_solve,
        ),
        solution,
    )
    return res
end

function solve_with_ma48(csc::SparseArrays.SparseMatrixCSC, rhs::Matrix)
    _t = time()
    solver = Ma48Solver(csc)
    t_init = time() - _t
    _t = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - _t
    sol = copy(rhs)
    _t = time()
    MadNLP.solve!(solver, sol)
    t_solve = time() - _t
    res = (;
        time = (;
            initialize = t_init,
            factorize = t_factorize,
            solve = t_solve,
        ),
        solution = sol,
    )
    return res
end

mutable struct BlockTriangularSolver
end

# TODO: What other information do I need for this solve
function solve!(csc::SparseArrays.SparseMatrixCSC, rhs::Matrix)
    # Compute BT form, including DAG
    solver = BlockTriangularSolver(csc)
    # Factorize diagonal blocks
    factorize!(solver)
    # Perform a block backsolve
    backsolve!(solver, rhs)
    return
end

IMAGE_INDEX = 7
ADVERSARIAL_LABEL = 1
THRESHOLD = 0.6

#nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
#nnfile = joinpath("nn-models", "mnist-relu1024nodes4layers.pt")
nnfile = joinpath("nn-models", "mnist-relu2048nodes4layers.pt")
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

nrhs = 10
RHS = ones(C_orig.m, nrhs)

CHECK_MA57 = true
if CHECK_MA57
    println("Solving regularized matrix with MA57")
    res_reg = solve_with_ma57(C_orig, RHS)
    println("Solving non-regularized matrix with MA57")
    res_noreg = solve_with_ma57(C, RHS)

    println("With reg.,    NNZ = $(SparseArrays.nnz(C_orig))")
    println("Without reg., NNZ = $(SparseArrays.nnz(C))")

    println("Factorize/solve times")
    println("---------------------")
    println("With regularization nonzeros:")
    display(res_reg.time)
    println("Without regularization nonzeros:")
    display(res_noreg.time)
end

# Here I'm making sure MA57 works with multiple RHS
#rhs = ones(C_orig.m, 3)
#solver = MadNLPHSL.Ma57Solver(C_orig)
#MadNLP.factorize!(solver)
#sol = copy(rhs)
#MadNLP.solve!(solver, sol)
#display(sol)

C_full = fill_upper_triangle(C)

println("Solving non-regularized system with MA48")
res_ma48 = solve_with_ma48(C_full, RHS)
println("Initialize/factorize/solve times")
println("--------------------------------")
display(res_ma48.time)

CHECK_UMFPACK = false
if CHECK_UMFPACK
    println("Solving non-regularized matrix with UMFPACK")
    res_umfpack = solve_with_umfpack(C_full, RHS)
    println("Factorize/solve times")
    println("---------------------")
    display(res_umfpack.time)
end

CHECK_BTF = true
if CHECK_BTF
    # Here, I'm making sure I can put C into BT order
    #println("Full symmetric pivot matrix:")
    #display(C_full)
    _t = time()
    igraph = MPIN.IncidenceGraphInterface(C_full)
    blocks = MPIN.block_triangularize(igraph)
    t_bt = time() - _t
    println("For reference, the time to compute BTF is $t_bt")
    row_order = vcat([cb for (cb, vb) in blocks]...)
    col_order = vcat([vb for (cb, vb) in blocks]...)
    C_bt = C_full[row_order, col_order]
    println("Pivot matrix in BT form")
    display(C_bt)
    
    lu = LinearAlgebra.lu(C_bt)
    rhs = ones(C_bt.m)
    x = lu \ rhs
    display(lu)
end
