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

# I might as well add this method to MathProgIncidence
function connected_components(matrix::Matrix)
    csc = SparseArrays.sparse(matrix)
    igraph = MathProgIncidence.IncidenceGraphInterface(csc)
    rowcc, colcc = MathProgIncidence.connected_components(igraph)
    # CCs don't have an inherent order, so we give them one based on
    # the minimum row index. TODO: This should be done by MathProgIncidence
    ncc = length(rowcc)
    order = sort(1:ncc; by = i -> minimum(rowcc[i]))
    rowcc = rowcc[order]
    colcc = colcc[order]
    return rowcc, colcc
end

struct BlockDiagonalLU
    row_partition::Vector{Vector{Int}}
    col_partition::Vector{Vector{Int}}
    factors::Vector{LinearAlgebra.LU}
end

function LinearAlgebra.lu(bd::BlockDiagonalView; check = true)
    # Update diagonal block matrices
    for (i, block) in enumerate(bd.blocks)
        block .= bd.matrix[bd.row_partition[i], bd.col_partition[i]]
    end
    factors = LinearAlgebra.lu.(bd.blocks; check)
    return BlockDiagonalLU(bd.row_partition, bd.col_partition, factors)
end

# TODO: I want to be able to factorize in-place. However, this may require me to update
# my BlockDiagonalLU struct.
#function LinearAlgebra.lu!(lu::BlockDiagonalLU; check = true)
#    # Update diagonal block matrices
#    for (i, block) in enumerate(bd.blocks)
#        block .= bd.matrix[bd.row_partition[i], bd.col_partition[i]]
#    end
#    factors = LinearAlgebra.lu.(bd.blocks; check)
#    return BlockDiagonalLU(bd.row_partition, bd.col_partition, factors)
#end

function LinearAlgebra.factorize(bd::BlockDiagonalView)
    return LinearAlgebra.lu(bd)
end

function LinearAlgebra.ldiv!(lu::BlockDiagonalLU, rhs::Vector)
    matrix = reshape(rhs, length(rhs), 1)
    LinearAlgebra.ldiv!(lu, matrix)
    rhs .= matrix
    return rhs
end

function LinearAlgebra.ldiv!(lu::BlockDiagonalLU, rhs::Matrix)
    rhs_blocks = map(indices -> rhs[indices, :], lu.row_partition)
    LinearAlgebra.ldiv!.(lu.factors, rhs_blocks)
    for (i, indices) in enumerate(lu.col_partition)
        rhs[indices, :] .= rhs_blocks[i]
    end
    return rhs
end
