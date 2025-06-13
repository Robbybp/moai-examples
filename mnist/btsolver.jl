import SparseArrays
import MathProgIncidence
import MadNLP

mutable struct BlockTriangularSolver
    csc::SparseArrays.SparseMatrixCSC
    blocks::Vector{Tuple{Vector{Int},Vector{Int}}}
    factors::Vector{Matrix}

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
    diagonal_block_matrices = map(b -> zeros(b, b), blocksizes)

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
    # - Partition rhs into blocks
    # - Backsolve blocks using stored LU factors (can probably store results in rhs)
    #   ^ I can't actually naively backsolve because RHSs are sequentially dependent
    csc = solver.csc
    blocks = solver.blocks
    factors = solver.factors
    nblock = length(blocks)
    nnz = SparseArrays.nnz(csc)
    # We partition the RHS by row blocks of the original matrix
    rhs_blocks = map(b -> view(rhs, b[1], :), blocks)

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

    # Backsolve using an adjacency list
    for b in 1:nblock
        solve!(factors[b], rhs_blocks[b])
        # Look up positions of this node's out-edges in edgelist
        for e in edgestart_by_block[b]:edgeend_by_block[b]
            _, j = dag[e]
            rhs_blocks[j] .-= off_diagonal_matrices[e] * rhs_blocks[b]
        end
    end
    dt = time() - _t
    println("[$dt] Backsolve")
    # By updating B in-place, we have implicitly applied the inverse
    # row permutation to our solution. We must undo this row permutation
    # *and* apply the column permutation to our solution in order to
    # recover the solution to the original problem.
    row_perm = [i for (rb, cb) in blocks for i in rb]
    # Maybe we need to apply the inverse column permutation here?
    col_perm = [j for (rb, cb) in blocks for j in cb]
    inv_col_perm = zeros(Int64, csc.n)
    for (i, j) in enumerate(col_perm)
        inv_col_perm[j] = i
    end
    rhs .= rhs[row_perm, :][inv_col_perm, :]
    dt = time() - _t
    println("[$dt] Reorder solution")
    return rhs
end
