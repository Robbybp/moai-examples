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

# Seems a little silly to extend MadNLP.factorize! here
function MadNLP.factorize!(solver::BlockTriangularSolver)
    csc = solver.csc
    blocks = solver.blocks
    # Extract diagonal matrices; factorize
    block_matrices = map(b -> Matrix(csc[b[1], b[2]]), blocks)
    # Doesn't seem worth it to parallelize here
    #nblocks = length(blocks)
    #nbatches = Threads.nthreads()
    #batchsize = Int(ceil(nblocks / nbatches))
    #batch_ranges = map(b -> (((b-1)*batchsize+1):min(nblocks, b*batchsize)), 1:nbatches)
    #
    # TODO: Vectorize these individual factorizations somehow?
    factorize!.(block_matrices)
    return
end
