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
    matrix::Matrix{Float64}
    blocks::Vector{Matrix{Float64}}
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

# This is mutable so I can update the `factors` field in-place.  I can't update
# the individual factors themselves in-place with lu!  because I'm using dense
# matrices (and LinearAlgebra seems not to support this). 
# TODO: Potentially use sparse matrices so I can use lu!(UmfpackLU, CSC)?
mutable struct BlockDiagonalLU
    row_partition::Vector{Vector{Int}}
    col_partition::Vector{Vector{Int}}
    factors::Vector{LinearAlgebra.LU{Float64,Matrix{Float64},Vector{Int}}}
end

# Method to instantiate an "empty" BlockDiagonalLU. `factors` is empty because
# we can't reuse these with lu! unless we switch to using sparse matrices.
BlockDiagonalLU(bd::BlockDiagonalView) = BlockDiagonalLU(bd.row_partition, bd.col_partition, [])

function LinearAlgebra.lu(bd::BlockDiagonalView; check = true)
    # Update diagonal block matrices
    for (i, block) in enumerate(bd.blocks)
        block .= bd.matrix[bd.row_partition[i], bd.col_partition[i]]
    end
    factors = LinearAlgebra.lu.(bd.blocks; check)
    return BlockDiagonalLU(bd.row_partition, bd.col_partition, factors)
end

function LinearAlgebra.lu!(lu::BlockDiagonalLU, bd::BlockDiagonalView; check = true)
    # Update diagonal block matrices
    for (i, block) in enumerate(bd.blocks)
        block .= bd.matrix[bd.row_partition[i], bd.col_partition[i]]
    end
    # This overwrites block matrices, but not the original matrix.
    factors = LinearAlgebra.lu!.(bd.blocks; check)
    lu.factors = factors
    return lu
end

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
