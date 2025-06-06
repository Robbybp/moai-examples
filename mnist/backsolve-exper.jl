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
include("btsolver.jl")

# TODO: Accept a matrix as the RHS
function solve_with_ma57(csc::SparseArrays.SparseMatrixCSC, rhs::Matrix)
    _t = time()
    solver = MadNLPHSL.Ma57Solver(csc)
    t_initialize = time() - _t
    _t = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - _t
    #println("INFO after factorization:")
    #display(collect(enumerate(solver.info)))
    sol = copy(rhs)
    _t = time()
    #println("INFO after backsolve:")
    #display(collect(enumerate(solver.info)))
    MadNLP.solve!(solver, sol)
    t_solve = time() - _t
    res = (;
        time = (;
            initialize = t_initialize,
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

function solve_with_ma48(
    csc::SparseArrays.SparseMatrixCSC,
    rhs::Matrix;
    min_blocksize::Int32 = Int32(1),
)
    nthreads = Threads.nthreads()
    if nthreads == 1
        _t = time()
        solver = Ma48Solver(csc; min_blocksize)
        t_init = time() - _t
        _t = time()
        MadNLP.factorize!(solver)
        t_factorize = time() - _t
        sol = copy(rhs)
        _t = time()
        MadNLP.solve!(solver, sol)
        t_solve = time() - _t
        t_total = t_init + t_factorize + t_solve
    else
        _t = time()
        sol = copy(rhs)
        rhsdim, nrhs = rhs.size
        println("nthreads = $nthreads")
        nbatches = nthreads
        # TODO: Make sure batchsize never gets too low
        batchsize = Int(ceil(nrhs / nbatches))
        println("batchsize = $batchsize")
        # Partition columns into batches. For each batch, we have a UnitRange of columns
        batch_columns = map(b -> (((b-1)*batchsize+1):min(nrhs, b*batchsize)), 1:nbatches)
        display(batch_columns)
        #batch_sols = [sol[:, batch_columns[b]] for b in 1:nbatches]
        Threads.@threads for b in 1:nbatches
            solver = Ma48Solver(csc)
            MadNLP.factorize!(solver)
            temp = sol[:, batch_columns[b]]
            MadNLP.solve!(solver, temp)
            sol[:, batch_columns[b]] = temp
        end
        #for b in 1:nbatches
        #    sol[:, batch_columns[b]] = batch_sols[b]
        #end
        t_total = time() - _t
        t_init = nothing
        t_factorize = nothing
        t_solve = nothing
    end
    res = (;
        time = (;
            initialize = t_init,
            factorize = t_factorize,
            solve = t_solve,
            total = t_total,
        ),
        solution = sol,
    )
    return res
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
nnfile = joinpath("nn-models", "mnist-relu1024nodes4layers.pt")
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
to_ignore = Set(Int(pivot_dim / 2 + 1):pivot_dim)
I, J, V = SparseArrays.findnz(C_orig)
to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
I = I[to_retain]
J = J[to_retain]
V = V[to_retain]
C = SparseArrays.sparse(I, J, V, C_orig.m, C_orig.n)

println("Decomposable pivot matrix:")
display(C)

nrhs = 100
RHS = ones(C_orig.m, nrhs)

CHECK_MA57 = false
if CHECK_MA57
    #println("Solving regularized matrix with MA57")
    #res_reg = solve_with_ma57(C_orig, RHS)
    println("Solving non-regularized matrix with MA57")
    res_noreg = solve_with_ma57(C, RHS)

    #println("With reg.,    NNZ = $(SparseArrays.nnz(C_orig))")
    println("Without reg., NNZ = $(SparseArrays.nnz(C))")

    println("Factorize/solve times")
    println("---------------------")
    #println("With regularization nonzeros:")
    #display(res_reg.time)
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

# What if my input matrix is already in block triangular form?
#var_indices = 1:Int(pivot_dim / 2)
#con_indices = Int(pivot_dim / 2 + 1):pivot_dim
#row_perm = vcat(con_indices, reverse(var_indices))
#col_perm = vcat(var_indices, reverse(con_indices))
#C_permuted = C_full[row_perm, col_perm]
#println("Permuted matrix to have close to block-triangular form")
#display(C_permuted)

CHECK_MA48 = false
if CHECK_MA48
    println("Solving non-regularized system with MA48")
    res_ma48 = solve_with_ma48(C_full, RHS; min_blocksize = Int32(128))
    println("Initialize/factorize/solve times")
    println("--------------------------------")
    display(res_ma48.time)
end

CHECK_UMFPACK = false
if CHECK_UMFPACK
    println("Solving non-regularized matrix with UMFPACK")
    res_umfpack = solve_with_umfpack(C_full, RHS)
    println("Factorize/solve times")
    println("---------------------")
    display(res_umfpack.time)
end

CHECK_BTF = false
if CHECK_BTF
    # Here, I'm making sure I can put C into BT order
    #println("Full symmetric pivot matrix:")
    #display(C_full)
    _t = time()
    igraph = MPIN.IncidenceGraphInterface(C_full)
    t_igraph = time() - _t
    _t = time()
    matching = MPIN.maximum_matching(igraph)
    t_matching = time() - _t
    _t = time()
    blocks = MPIN.block_triangularize(igraph)
    t_bt = time() - _t
    println("IncidenceGraphInterface times")
    println("-----------------------------")
    println("Initialize:    $t_igraph")
    println("Matching:      $t_matching")
    println("Block triang.: $t_bt")
    # Block triangularization is very expensive... It seems like
    # I should be able to make it faster.
    row_order = vcat([cb for (cb, vb) in blocks]...)
    col_order = vcat([vb for (cb, vb) in blocks]...)
    C_bt = C_full[row_order, col_order]
    println("Pivot matrix in BT form")
    display(C_bt)

    #lu = LinearAlgebra.lu(C_bt)
    #rhs = ones(C_bt.m)
    #x = lu \ rhs
    #display(lu)
end

_t = time()
btsolver = BlockTriangularSolver(C_full)
t_init = time() - _t
println("Initialize: $t_init")

_t = time()
MadNLP.factorize!(btsolver)
t_factorize = time() - _t
println("Factorize:  $t_factorize")

_t = time()
csc = btsolver.csc
blocks = btsolver.blocks
factors = btsolver.factors

# We partition the RHS by row blocks of the original matrix
rhs_blocks = map(b -> view(RHS, b[1], :), blocks)
nblocks = length(blocks)
# A couple options to populate dag:
# - quadratic loop over blocks
# - loop over NNZ
# TODO: cache these nz entries
# (Note that I can't cache V, so I'll need to run findnz here)
I, J, V = SparseArrays.findnz(csc)
nnz = SparseArrays.nnz(csc)

# TODO: Cache block maps on the BTSolver struct 
row_block_map = [(0,0) for _ in 1:csc.m]
col_block_map = [(0,0) for _ in 1:csc.n]
for (i, b) in enumerate(blocks)
    for (j, (r, c)) in enumerate(zip(b...))
        row_block_map[r] = (i, j)
        col_block_map[c] = (i, j)
    end
end
dt = time() - _t
println("[$dt] findnz and initialize block maps")

rowblock_by_nz = map(k -> row_block_map[I[k]][1], 1:nnz)
colblock_by_nz = map(k -> col_block_map[J[k]][1], 1:nnz)
dt = time() - _t
println("[$dt] Blocks by NZ arrays")

# This *is* the DAG. Instead of an adjacency list, it's an edgelist.
# I can sort this to get, basically, an adjacency list.
dag = collect(zip(colblock_by_nz, rowblock_by_nz))
#dag = map(k -> (col_block_map[J[k]][1], row_block_map[I[k]][1]), 1:nnz)
dt = time() - _t
println("[$dt] Build DAG with duplicate edges")
unique!(dag)
dt = time() - _t
println("[$dt] Filter duplicate entries")
filter!(e -> e[1] != e[2], dag)
dt = time() - _t
println("[$dt] Filter self-loops")
# Not sure if this will be necessary...
sort!(dag)
dt = time() - _t
println("[$dt] Sort DAG edges")

nedges = length(dag)

# This is fairly expensive (5 s), but up to this point, everything can be done
# in inialization.

# Filtering nonzeros is also expensive, but again, can be done in initialization
# This is actually so expensive (almost 4 s) that it might not be worth doing
#off_diagonal_nz = filter(k -> row_block_map[I[k]][1] != col_block_map[J[k]][1], 1:nnz)
off_diagonal_nz = filter(k -> rowblock_by_nz[k] != colblock_by_nz[k], 1:nnz)
nnz_offdiag = length(off_diagonal_nz)
dt = time() - _t
println("[$dt] Filter NZ")
# We sort by col block first because we will access these values first by
# col block, then by row block.

nrow = csc.m
ncol = csc.n
@assert nrow == ncol
nblock = length(blocks)

# This is expensive (about 4 s), but can also be moved to initialization.
# Without the extra indirection (from e.g., col_block_map[J[k]][1]), this can get
# heavily optimized and now only takes 1 s.
blockidx_by_nz = colblock_by_nz[off_diagonal_nz] * nblock .+ rowblock_by_nz[off_diagonal_nz]
off_diag_nzperm = sortperm(blockidx_by_nz)
dt = time() - _t
println("[$dt] Sort nonzero entries")
sorted_I = I[off_diagonal_nz][off_diag_nzperm]
sorted_J = J[off_diagonal_nz][off_diag_nzperm]
dt = time() - _t
println("[$dt] Sort I/J by (col block, row block)")

# This is the only thing so far that has to be done in the factorization
# or solve (when we have new matrix values).
sorted_V = V[off_diagonal_nz][off_diag_nzperm]
dt = time() - _t
println("[$dt] Sort V")

blockidx_by_sorted_nz = blockidx_by_nz[off_diag_nzperm]
# Now I semi-efficiently have the sorted nonzeros. I just need to get the start and
# end positions for each edge.
#
# These are positions in the sorted entries where either block changes
blockstarts = filter(k -> k == 1 || blockidx_by_sorted_nz[k] != blockidx_by_sorted_nz[k-1], 1:nnz_offdiag)
# If my math is correct, this has the same length as our edgelist
@assert nedges == length(blockstarts)
blockends = map(k -> blockstarts[k+1]-1, 1:(nedges-1))
push!(blockends, nnz_offdiag)
# Assumming we have sorted off-diagonal blocks correctly in both the edgelist and
# the nonzero indices, these should be in the same order.
# Now the question is: How do I want to store the off-diagonal matrices.
dt = time() - _t
println("[$dt] Partitioning nonzeros into off-diagonal blocks")

#off_diagonal = map(
#    # If edgedata exists primarily to construct these off-diagonal blocks,
#    # I just need it to contain the start and end indices
#    e -> (sorted_I[e[1]:e[2]], sorted_J[e[1]:e[2]], sorted_V[e[1];e[2]]),
#    # edgedata is an array of some imaginary data structure that contains
#    # the information I need
#    edgedata,
#)
slices = map(e -> (blockstarts[e]:blockends[e]), 1:nedges)
# edges are (colblock, rowblock), while blocks are (rows, cols)

# Map row/col coordinates to their positions-within-the-block
local_I = map(k -> row_block_map[sorted_I[k]][2], 1:nnz_offdiag)
local_J = map(k -> col_block_map[sorted_J[k]][2], 1:nnz_offdiag)
# I could also make these views
I_by_edge = map(s -> local_I[s], slices)
J_by_edge = map(s -> local_J[s], slices)
V_by_edge = map(s -> sorted_V[s], slices)
#I_views = map(s -> view(local_I, s), slices)
#J_views = map(s -> view(local_J, s), slices)
#V_views = map(s -> view(sorted_V, s), slices)
dt = time() - _t
println("[$dt] Construct local indices, by edge")

# These implementations are bad
#off_diagonal = map(e -> csc[blocks[e[2]][1], blocks[e[1]][2]], 1:nedges)
#off_diagonal = map(e -> SparseArrays.sparse(I_views[e], J_views[e], V_views[e]), 1:nedges)
# TODO: remove some of the indirection here
blocksizes = map(b -> length(b[1]), blocks)
rowblock_size_by_edge = map(e -> blocksizes[e[2]], dag)
colblock_size_by_edge = map(e -> blocksizes[e[1]], dag)

# How could this be broken down to avoid allocations in the backsolve?
# I have I/J by edge already.
# - csc comes in
# - I extract NZ with findnz -- is my life easier if I just sorted CSC directly?
#   findnz preserves row and nz order, so life is good
# - nzval gets sorted
# - I transfer these into a new array
# - V_by_edge, for each edge, is just an unsafe_wrap around part of this array
# - My sparse matrices are constructed using these unsafe_wraps, so they're updated
#   in-place -- How do I do this? sparse seems to copy its inputs by default
# - There must be some alternative to constructing an entire matrix just to set nzval
#   I think I can just use this with .=
#   This should be pretty fast
off_diagonal = map(
    e -> SparseArrays.sparse(
        I_by_edge[e], J_by_edge[e], V_by_edge[e], rowblock_size_by_edge[e], colblock_size_by_edge[e]
    ),
    1:nedges,
)
dt = time() - _t
println("[$dt] Construct sparse matrices to hold off-diagonal blocks")

for e in 1:nedges
    # There must be a way to do this with less overhead.
    off_diagonal[e].nzval .= V_by_edge[e]
end
dt = time() - _t
println("[$dt] Update nzval")

# What does my backsolve loop look like?
# Can I just loop over edges, or do I need the adjacency list?
iprev = 0
for (e, (i, j)) in enumerate(dag)
    # If this is the first time we've encountered i as a source node,
    # solve and store the solution in RHS
    if i != iprev
        solve!(factors[i], rhs_blocks[i])
    end
    # Used the cached solution X_i (in RHS_i) to update the RHS
    # of the destination node j
    rhs_blocks[j] .-= off_diagonal[e] * rhs_blocks[i]
end
dt = time() - _t
println("[$dt] Backsolve")

ADJACENCY_LIST = false
if ADJACENCY_LIST
    dag = [Int[] for _ in blocks]
    for k in 1:nnz
        push!(dag[col_block_map[J[k]][1]], row_block_map[I[k]][1])
    end
    # Building the adjacency list is also expensive (about 4 s)
    dt = time() - _t
    println("[$dt] Build DAG as a vector-of-vectors (adjacency list)")
    dag = map(unique, dag)
    dt = time() - _t
    println("[$dt] Filter duplicates from adjacency list")
end

#_t = time()
#solve!(btsolver, RHS)
#t_solve = time() - _t
#println("Solve:      $t_solve")
