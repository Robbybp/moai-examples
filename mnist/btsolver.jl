import SparseArrays
import MathProgIncidence
import MadNLP

mutable struct BlockTriangularSolver
    csc::SparseArrays.SparseMatrixCSC
    blocks::Vector{Tuple{Vector{Int},Vector{Int}}}
end

function BlockTriangularSolver(
    csc::SparseArrays.SparseMatrixCSC;
    max_blocksize::Int = 2,
)
    igraph = MathProgIncidence.IncidenceGraphInterface(csc)
    blocks = MathProgIncidence.block_triangularize(igraph)
    blocksizes = length.(blocks)
    if any(blocksizes .> max_blocksize)
        error(
            "Block triangular form has a block greater than the max blocksize of"
            * " $max_blocksize"
        )
    end
    return BlockTriangularSolver(csc, blocks)
end

function factorize!(matrix::Matrix)
    if matrix.size[1] == 1
        matrix[1, 1] = 1 / matrix[1, 1]
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

    # For each row/col, we store (a) the block it belongs to and (b) its position
    # within that block. While position-within-block is arbitrary, we will need to
    # keep this consistent
    row_block_map = [(0,0) for _ in 1:csc.m]
    col_block_map = [(0,0) for _ in 1:csc.n]
    for (i, b) in enumerate(blocks)
        for (j, (r, c)) in enumerate(zip(b...))
            row_block_map[r] = (i, j)
            col_block_map[c] = (i, j)
        end
    end

    # The following implementation, where I partition into 1x1 and 2x2 blocks,
    # then factorize each subset all at once, is only marginally faster than
    # a naive implementation. The dominant cost is extracting many small blocks
    # from the CSC matrix (either finding the values or building the matrix).
    # I should be able to get around this.
    #
    #d1_blocks = filter(b -> length(b[1]) == 1, blocks)
    #d2_blocks = filter(b -> length(b[2]) == 2, blocks)

    ## Presumably, we will need the matrices in order at some point
    ##block_matrices = map(b -> Matrix(csc[b[1], b[2]]), blocks)
    ##dt = time() - t0
    ##println("[$dt] Extracted block matrices")

    ## We just store 1x1 blocks as a single array
    #d1_matrices = map(b -> csc[b[1][1], b[2][1]], d1_blocks)
    #d2_matrices = map(b -> Matrix(csc[b[1], b[2]]), d2_blocks)
    #dt = time() - t0
    #println("[$dt] Partitioned into 1x1 and 2x2")

    #d2_array = Array{Float64,3}(undef, 2, 2, length(d2_matrices))
    #for i in 1:length(d2_matrices)
    #    d2_array[:, :, i] = d2_matrices[i]
    #end
    #dt = time() - t0
    #println("[$dt] Constructed 3d array")

    ## We just factorize the d1 matrices inline here
    #d1_matrices[:] ./= d1_matrices[:]
    #dt = time() - t0
    #println("[$dt] Factorized 1x1 matrices")
    #factorize_d2!(d2_array)
    #dt = time() - t0
    #println("[$dt] Factorized 2x2 matrices")

    # No real reason we need to update block_matrices in-place.
    # In fact, we could just construct this array here.
    #block_matrices[d1_blocks][1, 1] .= d1_matrices
    #block_matrices[d2_blocks][:, :] .= d2_matrices[:]

    ## Extract diagonal matrices; factorize
    #block_matrices = map(b -> Matrix(csc[b[1], b[2]]), blocks)
    ## TODO: Vectorize these individual factorizations somehow?
    #factorize!.(block_matrices)

    # Most of computational expense is in extracting the submatrices from `csc`.
    blocksizes = map(b -> length(first(b)), blocks)
    #d1_blocks = filter(b -> blocksizes[b] == 1, 1:length(blocks))
    #d2_blocks = filter(b -> blocksizes[b] == 2, 1:length(blocks))
    #d1_blocks = map(b -> blocks[b], d1_blocks)
    #d2_blocks = map(b -> blocks[b], d2_blocks)
    #d1_matrices = zeros(length(d1_blocks))
    #d2_matrices = zeros(2, 2, length(d2_blocks))

    #d1_row_block_map = [(0,0) for _ in 1:csc.m]
    #d1_col_block_map = [(0,0) for _ in 1:csc.n]
    #for (i, b) in enumerate(d1_blocks)
    #    for (j, (r, c)) in enumerate(zip(b...))
    #        d1_row_block_map[r] = (i, j)
    #        d1_col_block_map[c] = (i, j)
    #    end
    #end

    #d2_row_block_map = [(0,0) for _ in 1:csc.m]
    #d2_col_block_map = [(0,0) for _ in 1:csc.n]
    #for (i, b) in enumerate(d2_blocks)
    #    for (j, (r, c)) in enumerate(zip(b...))
    #        d2_row_block_map[r] = (i, j)
    #        d2_col_block_map[c] = (i, j)
    #    end
    #end
    #dt = time() - t0
    #println("[$dt] Block maps")

    # NOTE: It may be inefficient to store 1x1 blocks in a matrix
    block_matrices = map(b -> zeros(b, b), blocksizes)
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
        matrix = block_matrices[row_block_map[I[k]][1]]
        matrix[row_block_map[I[k]][2], col_block_map[J[k]][2]] += V[k]
    end

    # This implementation, where I split blocks into 1x1 and 2x2, appears to
    # be slightly slower than the alternative.
    #d1_block_entries = filter(k -> blocksizes[row_block_map[I[k]][1]] == 1, block_entries)
    #d2_block_entries = filter(k -> blocksizes[row_block_map[I[k]][1]] == 2, block_entries)
    ##for k in d1_block_entries
    ##    d1_matrices[d1_row_block_map[I[k]][1]] += V[k]
    ##end
    #d1_block_indices = map(k -> d1_row_block_map[I[k]][1], d1_block_entries)
    #d1_matrices[d1_block_indices] .+= V[d1_block_entries]
    #dt = time() - t0
    #println("[$dt] Populate d1 matrices from nonzeros")
    ##d2_row_indices = map(k -> d2_row_block_map[I[k]][2], d2_block_entries)
    ##d2_col_indices = map(k -> d2_col_block_map[J[k]][2], d2_block_entries)
    ##d2_block_indices = map(k -> d2_row_block_map[I[k]][1], d2_block_entries)
    ##d2_matrices[d2_row_indices, d2_col_indices, d2_block_indices] .+= V[d2_block_entries]
    #for k in d2_block_entries
    #    d2_matrices[d2_row_block_map[I[k]][2], d2_col_block_map[J[k]][2], d2_row_block_map[I[k]][1]] += V[k]
    #end
    #dt = time() - t0
    #println("[$dt] Populate d2 matrices from nonzeros")

    dt = time() - t0
    println("[$dt] Loop over nonzeros")
    factorize!.(block_matrices)
    #d1_matrices[:] ./= d1_matrices[:]
    #dt = time() - t0
    #println("[$dt] Factorized 1x1 matrices")
    #factorize_d2!(d2_matrices)
    dt = time() - t0
    println("[$dt] factorize")
    return
end
