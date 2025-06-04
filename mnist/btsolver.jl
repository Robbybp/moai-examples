import SparseArrays
import MathProgIncidence
import MadNLP

mutable struct BlockTriangularSolver
    csc::SparseArrays.SparseMatrixCSC
    blocks::Vector{Tuple{Vector{Int},Vector{Int}}}
    factors::Vector{Matrix}
end

function BlockTriangularSolver(
    csc::SparseArrays.SparseMatrixCSC;
    max_blocksize::Int = 2,
)
    igraph = MathProgIncidence.IncidenceGraphInterface(csc)
    blocks = MathProgIncidence.block_triangularize(igraph)
    blocksizes = map(b -> length(first(b)), blocks)
    if any(blocksizes .> max_blocksize)
        error(
            "Block triangular form has a block greater than the max blocksize of"
            * " $max_blocksize"
        )
    end
    # NOTE: It may be inefficient to store 1x1 blocks in a matrix, but this hasn't
    # shown up in a profile yet.
    block_matrices = map(b -> zeros(b, b), blocksizes)
    return BlockTriangularSolver(csc, blocks, block_matrices)
end

function factorize!(matrix::Matrix)
    if matrix.size[1] == 1
        # Do nothing
    elseif matrix.size[1] == 2
        # We update the 2x2 matrix to store its LU factors
        # 
        # | a11 a12 | = | l11     | | 1 u12 |
        # | a21 a22 |   | l21 l22 | |    1  |
        #
        # l11 = a11
        # u12 = a12 / l11
        # l21 = a21
        # l22 = a22 - l21 * u12
        #
        # We store:
        # | l11 u12 |
        # | l21 l22 |
        matrix[1, 2] /= matrix[1, 1]
        matrix[2, 2] -= matrix[2, 1] * matrix[1, 2]
    else
        error()
    end
    return
end

"""
Arguments
---------

* matrices: A 3-dimensional array containing n 2x2 matrices, where
            n is the dimension of the last axis.

"""
function factorize_d2!(matrices::Array{Float64,3})
#function factorize_d2!(matrices::Vector{Matrix{Float64}})
    matrices[1, 2, :] ./= matrices[1, 1, :]
    matrices[2, 2, :] .-= matrices[2, 1, :] .* matrices[1, 2, :]
    #matrices[:][1, 2] ./= matrices[:][1, 1]
    #matrices[:][2, 2] .-= matrices[:][2, 1] .* matrices[:][1, 2]
    return
end

# Seems a little silly to extend MadNLP.factorize! here
function MadNLP.factorize!(solver::BlockTriangularSolver)
    t0 = time()
    csc = solver.csc
    blocks = solver.blocks
    factors = solver.factors

    # For each row/col, we store (a) the block it belongs to and (b) its position
    # within that block. While position-within-block is arbitrary, we will need to
    # keep this consistent
    # This implementation is about 0.5 s faster, per factorization, than a
    # dict-based implementation on the 2048-by-4 NN.
    row_block_map = [(0,0) for _ in 1:csc.m]
    col_block_map = [(0,0) for _ in 1:csc.n]
    for (i, b) in enumerate(blocks)
        for (j, (r, c)) in enumerate(zip(b...))
            row_block_map[r] = (i, j)
            col_block_map[c] = (i, j)
        end
    end
    dt = time() - t0
    println("[$dt] Allocate matrices")
    I, J, V = SparseArrays.findnz(csc)
    dt = time() - t0
    println("[$dt] findnz")
    nnz = length(I)
    # Extract entries in the diagonal blocks. I.e., the row and column are
    # both in the same block.
    block_entries = filter(k -> row_block_map[I[k]][1] == col_block_map[J[k]][1], 1:nnz)
    # For entries in the same block, add the nonzero value to the matrix in the correct
    # position.
    for k in block_entries
        matrix = factors[row_block_map[I[k]][1]]
        matrix[row_block_map[I[k]][2], col_block_map[J[k]][2]] += V[k]
    end

    dt = time() - t0
    println("[$dt] Loop over nonzeros")
    factorize!.(factors)
    dt = time() - t0
    println("[$dt] factorize")
    return
end

function solve!(solver::BlockTriangularSolver, rhs::Vector)
    rhs = reshape(rhs, length(rhs), 1)
    return solve!(solver, rhs)
end

function solve!(lu::Matrix, rhs)
    # Note that we hit errors here if there's a divide-by-zero due to a bad pivot.
    # That may not be ideal...
    if lu.size[1] == 1
        rhs ./= lu[1, 1]
    elseif lu.size[1] == 2
        # lu stores both the L and U factors.
        #
        # lu = | l11 u12 |
        #      | l21 u22 |
        #
        # Our LU decomposition solves the following equations:
        #   
        #   l11 y1 = b1           (y1)
        #   l21 y1 + l22 y2 = b2  (y2)
        #   x2 = y2               (x2)
        #   x1 + u21 x2 = y1      (x1)
        #
        # Which is accomplished with the following in-place operations on RHS:
        #
        #   b1 <- ( b1 / l11         )
        #   b2 <- ( (b2 - l21) / l22 )
        #   b1 <- ( b1 - u21) b2     )
        rhs[1, :] .= rhs[1, :] ./ lu[1, 1]
        rhs[2, :] .= (rhs[2, :] .- lu[2, 1] .* rhs[1, :]) ./ lu[2, 2]
        rhs[1, :] .= rhs[1, :] - lu[1, 2] .* rhs[2, :]
    else
        error("Only diagonal blocks up to 2x2 are supported")
    end
end

function solve!(solver::BlockTriangularSolver, rhs::Matrix)
    _t = time()
    # - Partition rhs into blocks
    # - Backsolve blocks using stored LU factors (can probably store results in rhs)
    #   ^ I can't actually naively backsolve because RHSs are sequentially dependent
    csc = solver.csc
    blocks = solver.blocks
    factors = solver.factors
    # We partition the RHS by row blocks of the original matrix
    rhs_blocks = map(b -> view(rhs, b[1], :), blocks)
    nblocks = length(blocks)
    dag = [Int[] for _ in blocks]
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

    # TODO: All of this DAG construction can be done in initialization
    for k in 1:nnz
        push!(dag[col_block_map[J[k]][1]], row_block_map[I[k]][1])
    end
    dt = time() - _t
    println("[$dt] Build DAG")
    dag = map(unique, dag)
    nedges = sum(length, dag)
    # Global edgelist is 1:nedges, but I need the partition by source node
    # This is all making me think that a vector-of-vectors is not the right
    # data structure for my DAG.

    # Get a lookup from rowblock, colblock pairs to the index of the edge within
    # the colblock's adjacency list.
    # i is the rowblock index, j is the colblock index
    edge_indices = Dict(((i, j), e) for (j, adj) in enumerate(dag) for (e, i) in enumerate(adj))
    dt = time() - _t
    println("[$dt] Filter duplicates from DAG")
    blocksizes = map(b -> length(first(b)), blocks)
    off_diagonal = [map(i -> zeros(blocksizes[i], blocksizes[j]), adj) for (j, adj) in enumerate(dag)]
    off_diagonal_nz = filter(k -> row_block_map[I[k]][1] != col_block_map[J[k]][1], 1:nnz)
    dt = time() - _t
    println("[$dt] Allocate matrices and filter NZs")

    # I'd like an approach that iterates over the DAG's edges, but that doesn't have
    # to extract CSC indices for each edge.
    off_diagonal = map(
        # If edgedata exists primarily to construct these off-diagonal blocks,
        # I just need it to contain the start and end indices
        e -> (sorted_I[e[1]:e[2]], sorted_J[e[1]:e[2]], sorted_V[e[1];e[2]]),
        # edgedata is an array of some imaginary data structure that contains
        # the information I need
        edgedata,
    )
    # I'll also need to know which edge corresponds to a particular DAG entry.
    # A data structure with the same shape as `dag` that contains indices into
    # the global edge list.

    # This approach doesn't seem horrible. In some sense, doing a single
    # loop over nonzero values is the best I can do.
    # How does this look in an imaginary scenario where I don't need dict lookup?
    for k in off_diagonal_nz
        row_block, row_idx = row_block_map[I[k]]
        col_block, col_idx = col_block_map[J[k]]
        #edge_idx = edge_indices[row_block, col_block]
        off_diagonal[col_block][1][row_idx, col_idx] += V[k]
    end
    #off_diagonal = [map(i -> csc[blocks[i][1], blocks[j][2]], adj) for (j, adj) in enumerate(dag)]
    dt = time() - _t
    println("[$dt] Populate block matrices with NZ values")

    for i in 1:nblocks
        solve!(factors[i], rhs_blocks[i])
        # Update all other RHS blocks that this one depends on
        for (e, j) in enumerate(dag[i])
            # e is the "edge index" connecting i to j in the DAG
            # j is the index of the row block
            rhs_blocks[j] .-= off_diagonal[i][e] * rhs_blocks[i]
        end
    end
    dt = time() - _t
    println("[$dt] Backsolve and update RHS")
    return rhs
end
