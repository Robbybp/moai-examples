import BlockDiagonals
import LinearAlgebra
import MathProgIncidence
import SparseArrays

"""
For my current purposes, I'm better served by a wrapper around an existing
dense matrix that extracts values from diagonal blocks for factorization
and backsolve. I think this will greatly simplify the process of updating
values pre-factorization.
"""
struct BlockDiagonalView
    matrix::Matrix
    blocks::Vector{Matrix}
    row_partition::Vector{Vector{Int}}
    col_partition::Vector{Vector{Int}}
    function BlockDiagonalView(
        matrix::Matrix,
        row_partition::Vector{Vector{Int}},
        col_partition::Vector{Vector{Int}},
    )
        @assert length(row_partition) == length(col_partition)
        @assert all(length.(row_partition) .== length.(col_partition))
        nblocks = length(row_partition)
        blocks = map(i -> matrix[row_partition[i], col_partition[i]], 1:nblocks)
        return new(matrix, blocks, row_partition, col_partition)
    end
end

"""
    BlockDiagonal(matrix)

Convert a matrix that is already in block-diagonal order
to a `BlockDiagonal` matrix.

"""
function BlockDiagonals.BlockDiagonal(matrix::Matrix, blocksizes::Vector{Int})
    i = 1
    blocks = []
    for n in blocksizes
        indices = i:(i + n - 1)
        push!(blocks, matrix[indices, indices])
        i += n
    end
    return BlockDiagonals.BlockDiagonal(blocks)
    #csc = SparseArrays.sparse(matrix)
    #igraph = MathProgIncidence.IncidenceGraphInterface(csc)
    #rowcc, colcc = MathProgIncidence.connected_components(igraph)
    ## rowcc and colcc partition the rows and columns, but I don't
    ## guarantee anything about their order.
    ## Conditions required for the matrix to be block-diagonal:
    ## - Each CC contains contiguous row and col indices
    ## Then I just sort the CCs and sort the indices within them,
    ## and those are the diagonal blocks.
    #@assert length(rowcc) == length(colcc)
    #ncc = length(rowcc)
    #sort!.(rowcc)
    #sort!.(colcc)
    #ccs_contiguous = all(Set.(rowcc) == Set(first(rowcc):last(rowcc)))
    #rcc_order = sort(1:ncc, by = cc -> first(cc))
    #ccc_order = sort(1:ncc, by = cc -> first(cc))
    #rowsizes = length.(concc)
    #colsizes = length.(varcc)
    #if all(Set.(rowcc) == )
    #end
end

# I might as well add this method to MathProgIncidence
function connected_components(matrix::Matrix)
    csc = SparseArrays.sparse(matrix)
    igraph = MathProgIncidence.IncidenceGraphInterface(csc)
    rowcc, colcc = MathProgIncidence.connected_components(igraph)
    return rowcc, colcc
    # It is easy to get the permutation and the blocksizes from these
    # partitions.
end

struct BlockDiagonalLU
    index_partition::Vector{UnitRange}
    factors::Vector{LinearAlgebra.LU}
    function BlockDiagonalLU(bm::BlockDiagonals.BlockDiagonal)
        index_partition = []
        blocksizes = BlockDiagonals.blocksizes(bm)
        start = 1
        for (nrow, ncol) in blocksizes
            @assert nrow == ncol
            push!(index_partition, start:(start + nrow - 1))
            start += nrow
        end
        # TODO: use `lu!`? This modifies bm in-place, which should be fine
        factors = LinearAlgebra.lu(BlockDiagonals.blocks(bm))
        return BlockDiagonalLU(factors)
    end
end

function LinearAlgebra.lu(bm::BlockDiagonals.BlockDiagonal)
    return BlockDiagonalLU(bm)
end

function LinearAlgebra.factorize(bm::BlockDiagonals.BlockDiagonal)
    return BlockDiagonalLU(bm)
end

function LinearAlgebra.ldiv!(lu::BlockDiagonalLU, rhs::Vector)
    matrix = reshape(rhs, length(rhs), 1)
    LinearAlgebra.ldiv!(lu, matrix)
    rhs .= matrix
    return rhs
end

function LinearAlgebra.ldiv!(lu::BlockDiagonalLU, rhs::Matrix)
    rhs_blocks = map(indices -> rhs[indices, :], lu.index_partition)
    ldiv!.(lu.factors, rhs_blocks)
    rhs[indices] .= rhs_blocks[i]
    return rhs
end
