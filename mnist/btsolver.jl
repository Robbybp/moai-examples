import SparseArrays
import LinearAlgebra
import MathProgIncidence
import MadNLP
# I'm currently not planning to use BlockDiagonals
#import BlockDiagonals

include("blockdiagonal.jl")

mutable struct BlockTriangularSolver
    csc::SparseArrays.SparseMatrixCSC
    blocks::Vector{Tuple{Vector{Int},Vector{Int}}}

    # Data structures for factorization
    diagonal_block_matrices::Vector{Matrix}
    blockdiagonal_views::Vector{BlockDiagonalView}
    # Indices, within `diagonal_block_matrices`, of matrices that we will
    # block-diagonalize. (or should I just block-diagonalize all of these
    # matrices?)
    blockdiagonal_indices::Vector{Int}
    #factors::Vector{<:LinearAlgebra.Factorization}
    # Using Vector{Any} here because LinearAlgebra.factorize doesn't
    # appear to be type-stable.
    #
    # TODO: Custom factorization object for block-diagonal matrices
    # Or, I try to store the factors in-place. But this breaks down
    # when my blocks are themselves block-triangular rather than
    # block-diagonal.
    # ^ Actually, I think this is fine. The factor of an LT matrix
    # is itself. It only breaks down for *block* LT matrices.
    #
    # - If the matrix is diagonal, use built-in `factorize`
    # - If the matrix is LT, use built-in `factorize`
    # - If the matrix is block-diagonal, use built-in factorize on
    #   diagonal blocks?
    #
    # What does this look like now? I would like to store either
    # Matrix or BlockDiagonal for each diagonal block. So the above
    # also needs to be `Vector{Any}`. Then I will simply call
    # factorize.(diagonal_block_matrices).
    factors::Vector{Any}

    # The following is the minimal data I need for the backsolve
    off_diagonal_nz::Vector{Int}
    # Permutation that sorts nonzeros by edge in the DAG
    off_diagonal_nzperm::Vector{Int}
    # Partitions nonzeros by edge
    edge_slices::Vector{UnitRange{Int}}
    nedges::Int
    # This is the edge list.
    dag::Vector{Tuple{Int, Int}}
    # These define an adjacency list. They are indices in the edge list
    # corresponding to this node's out-edges
    edgestart_by_block::Vector{Int}
    edgeend_by_block::Vector{Int}
    off_diagonal_matrices::Vector{SparseArrays.SparseMatrixCSC}
end

function BlockTriangularSolver(
    csc::SparseArrays.SparseMatrixCSC;
    blocks::Union{Vector,Nothing} = nothing,
    #max_blocksize::Int = 2,
)
    @assert csc.m == csc.n
    dim = csc.m
    igraph = MathProgIncidence.IncidenceGraphInterface(csc)

    # TODO: If `blocks` is provided, raise an error if the blocks don't partition
    # `csc`'s rows and columns.
    if blocks === nothing
        blocks = MathProgIncidence.block_triangularize(igraph)
    else
        # If blocks were provided, we need to make sure they partition the row/column indices
        # I also should make sure they give an LT ordering. TODO.
        all_row_indices = mapreduce(b -> b[1], vcat, blocks)
        all_col_indices = mapreduce(b -> b[2], vcat, blocks)
        expected_indices = Set(1:dim)
        @assert length(all_row_indices) == dim
        @assert length(all_col_indices) == dim
        @assert issubset(all_row_indices, expected_indices)
        @assert issubset(all_col_indices, expected_indices)
    end
    blocksizes = map(b -> length(first(b)), blocks)

    # Here, I must determine how to store each matrix. For now, the options
    # are `Matrix` or `BlockDiagonal`. I need to use a custom type in order
    # to use the correct `factorize` method.
    # 
    # The problem is that I don't actually have the matrices explicitly here.
    # Extracting dense matrices is unreliable due to explicit zeros.
    csc_blocks = map(b -> csc[b...], blocks)
    # Once I have a sparse matrix for every block:
    block_ccs = MathProgIncidence.connected_components.(csc_blocks)
    # Block-diagonal blocksizes
    bd_blocksizes = map(ccs -> map(cc -> length(cc), ccs[1]), block_ccs)
    # What is my criteria for using BlockDiagonal?
    #use_block_diagonal = filter(
    #    # We use BlockDiagonal if the block-diagonal blocksize is no more than
    #    # 10% of the block-triangular blocksize.
    #    i -> maximum(bd_blocksizes[i]) / blocksizes[i] <= 0.1,
    #    1:length(blocks),
    #)
    use_block_diagonal = collect(1:length(blocks))
    # TODO: Should I also store the indices for which we *don't* use block diagonalization?

    diagonal_block_matrices = map(b -> zeros(b, b), blocksizes)
    blockdiagonal_views = map(
        i -> BlockDiagonalView(diagonal_block_matrices[i], block_ccs[i]...),
        use_block_diagonal,
    )
    #diagonal_block_matrices[use_block_diagonal] .= BlockDiagonalView.(
    #    diagonal_block_matrices[use_block_diagonal],
    #    bd_blocksizes[use_block_diagonal],
    #)

    # This will contain the outputs of LinearAlgebra.factorize, which is not
    # type-stable.
    factors = Any[]

    nblock = length(blocks)
    nnz = SparseArrays.nnz(csc)
    I, J, _ = SparseArrays.findnz(csc)

    # TODO: row/col_block_map and row/col_offset_map should probably
    # be separate arrays
    row_block_map = [(0,0) for _ in 1:csc.m]
    col_block_map = [(0,0) for _ in 1:csc.n]
    for (i, b) in enumerate(blocks)
        for (j, (r, c)) in enumerate(zip(b...))
            row_block_map[r] = (i, j)
            col_block_map[c] = (i, j)
        end
    end
    rowblock_by_nz = map(k -> row_block_map[I[k]][1], 1:nnz)
    colblock_by_nz = map(k -> col_block_map[J[k]][1], 1:nnz)
    # Note that this is doing basically the same filter that I do when constructing
    # the DAG below.
    off_diagonal_nz = filter(k -> rowblock_by_nz[k] != colblock_by_nz[k], 1:nnz)
    nnz_offdiag = length(off_diagonal_nz)

    # In our DAG, column blocks are source nodes and row blocks are destination
    # nodes.
    # TODO: Does this get faster if I store source and destination nodes in
    # separate arrays?
    # I would need to use unique(e -> (colblock_by_nz[e], rowblock_by_nz[e])),
    # but this would likely reduce the amount of indexing elsewhere
    dag = collect(zip(colblock_by_nz, rowblock_by_nz))
    unique!(dag)
    filter!(e -> e[1] != e[2], dag)
    # Sorting entries by (colblock, rowblock) puts the edges in topological
    # order (in some sense).
    sort!(dag)
    nedges = length(dag)

    # "blockidx" is a unique integer index for each off-diagonal block. Blocks are sorted
    # first by column block, then by row block.
    blockidx_by_nz = colblock_by_nz[off_diagonal_nz] * nblock .+ rowblock_by_nz[off_diagonal_nz]
    off_diagonal_nzperm = sortperm(blockidx_by_nz)
    sorted_I = I[off_diagonal_nz][off_diagonal_nzperm]
    sorted_J = J[off_diagonal_nz][off_diagonal_nzperm]

    # Can I just iterate over sorted_I/J directly here?
    local_I = map(i -> row_block_map[i][2], sorted_I)
    local_J = map(j -> col_block_map[j][2], sorted_J)

    # Note that, here, I'm constructing basically the exact same DAG I construct above.
    # The only difference is that I extract extra info: the block start/end-points in the
    # space of sorted nonzeros. I.e., there is a fairly significant amount of duplicated
    # work.
    blockidx_by_sorted_nz = blockidx_by_nz[off_diagonal_nzperm]
    blockstarts = filter(k -> k == 1 || blockidx_by_sorted_nz[k] != blockidx_by_sorted_nz[k-1], 1:nnz_offdiag)
    @assert nedges == length(blockstarts)
    blockends = map(k -> blockstarts[k+1]-1, 1:(nedges-1))
    push!(blockends, nnz_offdiag)

    # These are the nonzeros corresponding to each off-diagonal block
    edge_slices = map(e -> (blockstarts[e]:blockends[e]), 1:nedges)
    I_by_edge = map(s -> local_I[s], edge_slices)
    J_by_edge = map(s -> local_J[s], edge_slices)

    zeros_by_edge = map(e -> zeros(blockends[e] + 1 - blockstarts[e]), 1:nedges)
    # TODO: remove some of the indirection here
    blocksizes = map(b -> length(b[1]), blocks)
    rowblock_size_by_edge = map(e -> blocksizes[e[2]], dag)
    colblock_size_by_edge = map(e -> blocksizes[e[1]], dag)
    off_diagonal_matrices = map(
        e -> SparseArrays.sparse(
            I_by_edge[e], J_by_edge[e], zeros_by_edge[e], rowblock_size_by_edge[e], colblock_size_by_edge[e]
        ),
        1:nedges,
    )

    # Here we are constructing an adjacency list from the sorted edge list
    # Initialize each block to hold an empty set of contiguous indices
    edgestart_by_block = ones(Int64, nblock)
    edgeend_by_block = zeros(Int64, nblock)
    nonsink_edgestarts = unique(e -> dag[e][1], 1:nedges)
    nonsink_blocks = first.(dag[nonsink_edgestarts])
    edgestart_by_block[nonsink_blocks] .= nonsink_edgestarts
    for (b, e) in zip(nonsink_blocks, nonsink_edgestarts)
        edgeend = edgestart_by_block[b] - 1
        # Look ahead one edge and move forward if the next edge has the same
        # source node.
        while edgeend+1 <= nedges && dag[edgeend+1][1] == b
            edgeend += 1
        end
        edgeend_by_block[b] = edgeend
    end

    return BlockTriangularSolver(
        csc,
        blocks,
        diagonal_block_matrices,
        blockdiagonal_views,
        use_block_diagonal,
        factors,
        off_diagonal_nz,
        off_diagonal_nzperm,
        edge_slices,
        nedges,
        dag,
        edgestart_by_block,
        edgeend_by_block,
        off_diagonal_matrices,
    )
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

function factorize!(solver::BlockTriangularSolver)
    t0 = time()
    csc = solver.csc
    blocks = solver.blocks
    block_matrices = solver.diagonal_block_matrices

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
    #
    # FIXME: There is a bug here regarding repeated solves. I'm assuming that these
    # matrices are initialized to zeros, which is clearly not the case as written.
    for k in block_entries
        matrix = block_matrices[row_block_map[I[k]][1]]
        matrix[row_block_map[I[k]][2], col_block_map[J[k]][2]] += V[k]
    end

    dt = time() - t0
    println("[$dt] Loop over nonzeros")
    # Note that this allocates new Factorization objects.
    # Note that if I'm only block-diagonalizing a subset of the blocks, I need
    # to call factorize in two rounds: Once for the raw matrices and once for
    # the `BlockDiagonalView`s.
    factors = LinearAlgebra.factorize.(solver.blockdiagonal_views)
    solver.factors = factors
    dt = time() - t0
    println("[$dt] factorize")
    return
end

function solve!(solver::BlockTriangularSolver, rhs::Vector)
    rhs = reshape(rhs, length(rhs), 1)
    return solve!(solver, rhs)
end

function solve!(lu::Matrix{Float64}, rhs)
    # Note that we hit errors here if there's a divide-by-zero due to a bad pivot.
    # That may not be ideal...
    if lu.size[1] == 1
        rhs ./= lu[1, 1]
    elseif lu.size[1] == 2
        # lu stores both the L and U factors.
        #
        # lu = | l11 u12 |
        #      | l21 l22 |
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
        #
        rhs[1, :] .= rhs[1, :] ./ lu[1, 1]
        rhs[2, :] .= (rhs[2, :] .- lu[2, 1] .* rhs[1, :]) ./ lu[2, 2]
        rhs[1, :] .= rhs[1, :] - lu[1, 2] .* rhs[2, :]
    else
        error("Only diagonal blocks up to 2x2 are supported")
    end
end

function solve!(solver::BlockTriangularSolver, rhs::Matrix)
    _t = time()
    csc = solver.csc
    blocks = solver.blocks
    #diagonal_block_matrices = solver.diagonal_block_matrices
    factors = solver.factors
    nblock = length(blocks)
    nnz = SparseArrays.nnz(csc)

    row_perm = [i for (rb, cb) in blocks for i in rb]
    col_perm = [j for (rb, cb) in blocks for j in cb]
    inv_col_perm = zeros(Int64, csc.n)
    for (i, j) in enumerate(col_perm)
        inv_col_perm[j] = i
    end

    # We partition the RHS by row blocks of the original matrix
    #rhs_blocks = map(b -> view(rhs, b[1], :), blocks)
    # `rhs_blocks` now contains a copy of rhs
    rhs_blocks = map(b -> rhs[b[1], :], blocks)

    off_diagonal_nz = solver.off_diagonal_nz
    off_diagonal_nzperm = solver.off_diagonal_nzperm
    edge_slices = solver.edge_slices
    nedges = solver.nedges
    dag = solver.dag
    edgestart_by_block = solver.edgestart_by_block
    edgeend_by_block = solver.edgeend_by_block
    off_diagonal_matrices = solver.off_diagonal_matrices

    I, J, V = SparseArrays.findnz(csc)
    sorted_V = V[off_diagonal_nz][off_diagonal_nzperm]
    V_by_edge = map(s -> sorted_V[s], edge_slices)

    for e in 1:nedges
        # There must be a way to do this with less overhead.
        # I would like each matrix's nzval to be an unsafe wrap around
        # part of some global array. Not sure if this is possible.
        # This also gets much more efficient when I use fewer blocks.
        off_diagonal_matrices[e].nzval .= V_by_edge[e]
    end
    dt = time() - _t
    println("[$dt] Update nzval")

    # What happens if I use dense off-diagonal blocks?
    #off_diagonal_matrices = map(m -> Matrix(m), off_diagonal_matrices)
    off_diagonal_matrices = map(
        # Some quick heuristic to switch between sparse and dense?
        matrix -> SparseArrays.nnz(matrix) >= 0.01*matrix.m*matrix.n ? Matrix(matrix) : matrix,
        off_diagonal_matrices,
    )
    dt = time() - _t
    println("[$dt] Construct dense matrices")

    # Backsolve using an adjacency list
    t_solve = 0.0
    t_loop = 0.0
    t_multiply = 0.0
    t_subtract = 0.0
    _t = time()
    println()
    println("Entering backsolve loop for $nblock blocks and $nedges edges")
    for b in 1:nblock
        local _t = time()
        #solve!(diagonal_block_matrices[b], rhs_blocks[b])
        # What ever struct I return from `factorize`, it should support
        # ldiv!. This way I can return Julia built-in factors as well
        # if they're convenient and performant.
        LinearAlgebra.ldiv!(factors[b], rhs_blocks[b])
        t_solve += time() - _t
        # Look up positions of this node's out-edges in edgelist
        # TODO: It may be worth stacking the matrices corresponding to out-edges
        # and doing all of this in one multiplication + subtraction. This is possible
        # because all out-edges can be processed independently.
        for e in edgestart_by_block[b]:edgeend_by_block[b]
            _, j = dag[e]
            local _t = time()
            #rhs_blocks[j] .-= off_diagonal_matrices[e] * rhs_blocks[b]
            temp = off_diagonal_matrices[e] * rhs_blocks[b]
            dt = time() - _t
            t_multiply += dt
            rhs_blocks[j] .-= temp
            t_subtract += time() - _t - dt
            #println("Subtracting the product for edge $e, between nodes $b and $j, took $(@sprintf("%1.1f", dt)) s")
            #display(SparseArrays.sparse(off_diagonal_matrices[e]))
        end
    end
    t_loop = time() - _t
    println("---------------")
    println("Backsolve loop:")
    println("Solve:    $t_solve")
    println("Mutiply:  $t_multiply")
    println("Subtract: $t_subtract")
    println("Other:    $(t_loop-t_solve-t_multiply-t_subtract)")
    println("Total:    $t_loop")
    println("---------------")
    dt = time() - _t
    println("[$dt] Backsolve")
    for (i, rhs_i) in enumerate(rhs_blocks)
        rhs[blocks[i][1], :] .= rhs_i
    end
    dt = time() - _t
    println("[$dt] Update rhs")
    # By updating B in-place, we have implicitly applied the inverse
    # row permutation to our solution. We must undo this row permutation
    # *and* apply the column permutation to our solution in order to
    # recover the solution to the original problem.
    rhs .= rhs[row_perm, :][inv_col_perm, :]
    dt = time() - _t
    println("[$dt] Reorder solution")
    return rhs
end
