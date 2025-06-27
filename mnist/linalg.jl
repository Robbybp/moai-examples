import SparseArrays
import LinearAlgebra
import MadNLP
import MadNLPHSL
import HSL

# For some reason, we need to extend this method
MadNLP.parse_option(::Any, vec::Vector) = vec

function MadNLP.solve!(M::MadNLPHSL.Ma57Solver{T,INT}, rhs::Matrix{T}) where {T,INT}
    rhsdim, nrhs = rhs.size
    lwork = INT(rhsdim * nrhs)
    # TODO: allocate work array in initialization.
    work = Vector{T}(undef, lwork)
    HSL.ma57cr(
        T,
        INT,
        one(INT),
        INT(M.csc.n),
        M.fact,
        M.lfact,
        M.ifact,
        M.lifact,
        INT(nrhs),
        # Sending the RHS array directly to MA57 seems to work.
        # Both arrays are column-major.
        rhs,
        # LRHS. This is the matrix dimension, not NRHS
        INT(M.csc.n),
        work,
        lwork,
        M.iwork,
        M.icntl,
        M.info,
    )
    #display(collect(enumerate(M.info)))
    M.info[1] < 0 && throw(MadNLPHSL.SolveException())
    return rhs
end

function MadNLP.solve!(M::MadNLP.AbstractLinearSolver, rhs::Matrix)
    _, nrhs = rhs.size
    for j in 1:nrhs
        temp = rhs[:, j]
        MadNLP.solve!(M, temp)
        rhs[:, j] = temp
    end
    return rhs
end

mutable struct SchurComplementFactorizeTimer
    total::Float64
    reduced::Float64
    pivot::Float64
    update_pivot::Float64
    solve::Float64
    multiply::Float64
    update_reduced::Float64
    SchurComplementFactorizeTimer() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

mutable struct SchurComplementTimer
    initialize::Float64
    factorize::SchurComplementFactorizeTimer
    solve::Float64
    SchurComplementTimer() = new(0.0, SchurComplementFactorizeTimer(), 0.0)
end

function Base.show(io::IO, timer::SchurComplementTimer)
    println(io, "SchurComplementSolver timing information")
    println(io, "----------------------------------------")
    println(io, "initialize: $(timer.initialize)")
    println(io, "factorize:  $(timer.factorize.total)")
    println(io, "  reduced:        $(timer.factorize.reduced)")
    println(io, "  pivot:          $(timer.factorize.pivot)")
    println(io, "  update_pivot:   $(timer.factorize.update_pivot)")
    println(io, "  solve:          $(timer.factorize.solve)")
    println(io, "  multiply:       $(timer.factorize.multiply)")
    println(io, "  update_reduced: $(timer.factorize.update_reduced)")
    other = (
        timer.factorize.total
        - timer.factorize.reduced
        - timer.factorize.pivot
        - timer.factorize.update_pivot
        - timer.factorize.solve
        - timer.factorize.multiply
        - timer.factorize.update_reduced
    )
    println(io, "  other:          $(other)")
    println(io, "solve:      $(timer.solve)")
    println(io, "----------------------------------------")
end

mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions
    ReducedSolver::Type
    PivotSolver::Type
    pivot_indices::Vector{INT}
    # TODO: Vector{Tuple{Vector{INT},Vector{INT}}}. I.e., use same int type.
    # TODO: Move pivot_index_partition to BTSolverOptions or something.
    pivot_index_partition::Union{Nothing,Vector}
    function SchurComplementOptions(;
        # TODO: Non-third party default subsolver
        ReducedSolver = MadNLPHSL.Ma27Solver,
        PivotSolver = MadNLPHSL.Ma27Solver,
        pivot_indices = Int32[],
    ) 
        return new{eltype(pivot_indices)}(
            ReducedSolver,
            PivotSolver,
            pivot_indices,
            nothing,
        )
    end
end

struct SchurComplementSolver{T,INT} <: MadNLP.AbstractLinearSolver{T}
    csc::SparseArrays.SparseMatrixCSC{T,INT}
    reduced_solver::MadNLP.AbstractLinearSolver{T}
    pivot_solver::MadNLP.AbstractLinearSolver{T}
    # These are indices on which we pivot rows and columns.
    pivot_indices::Vector{INT}
    timer::SchurComplementTimer
end

function _sparse_schur(
    csc::SparseArrays.SparseMatrixCSC,
    pivot_indices::Vector;
    linear_solver = nothing,
)
    dim = csc.n
    pivot_dim = length(pivot_indices)
    reduced_dim = dim - pivot_dim
    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)
    P = pivot_indices
    R = reduced_indices
    A = csc[R, R]
    # We need the off-diagonal block in the *permuted* matrix.
    # For entries with p<r in the original matrix, we must transpose the indices.
    B = csc[P, R] + csc[R, P]'
    C = csc[P, P]
    B_dense = Matrix(B)
    sol = copy(B_dense)
    if linear_solver === nothing
        linear_solver = MadNLPHSL.Ma27Solver(C)
        MadNLP.factorize!(linear_solver)
    end
    for j in 1:reduced_dim
        temp = sol[:, j]
        # view(sol, :, j) isn't working here, even though it seems like it should...
        MadNLP.solve!(linear_solver, temp)
        sol[:, j] = temp
    end
    # Converting to sparse here removes explicit zeros
    term2 = B' * SparseArrays.sparse(sol)
    schur_complement = A - LinearAlgebra.tril(term2)
    return schur_complement
end

function SchurComplementSolver(
    csc::SparseArrays.SparseMatrixCSC{T,INT};
    opt::SchurComplementOptions = SchurComplementOptions(),
    logger::MadNLP.MadNLPLogger = MadNLP.MadNLPLogger(),
    ReducedSolver::Type = opt.ReducedSolver,
    PivotSolver::Type = opt.PivotSolver,
    pivot_indices::Vector{INT} = opt.pivot_indices,
    pivot_index_partition = opt.pivot_index_partition,
) where {T,INT}
    FloatType = eltype(csc.nzval)
    IntType = eltype(csc.rowval)
    timer = SchurComplementTimer()
    t_start = time()
    # TODO: I need to make sure these submatrices get updated when CSC changes
    # To do this, I will need to construct the submatrices directly from the
    # nz array of the original matrix.
    @assert csc.n == csc.m
    dim = csc.n
    pivot_dim = length(pivot_indices)
    reduced_dim = dim - pivot_dim
    Iorig, Jorig, Vorig = SparseArrays.findnz(csc)
    # Assert that only lower triangular entries are provided.
    # If parts of the upper triangle are provided, our Schur complement below
    # will be incorrect.
    @assert all(Iorig .>= Jorig)

    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)
    R = reduced_indices
    P = pivot_indices

    # The KKT matrix is:
    # | A B^T |
    # | B  C  |
    #
    # The Schur complement is:
    #   A - B^T C^-1 B
    #
    A = csc[R, R]
    # We have to extract coordinates from B and B^T because we are not necessarily
    # given the lower triangle of the KKT matrix in the order we want (R, P).
    B = csc[P, R] + csc[R, P]'

    # Allocate a matrix containing all possible nonzeros for the reduced matrix
    #
    # Here, we assume this is a dense matrix:
    #I = IntType[i for i in 1:reduced_dim for j in 1:reduced_dim]
    #J = IntType[j for i in 1:reduced_dim for j in 1:reduced_dim]
    #V = FloatType[0.0 for _ in I]
    #
    # Here, we exploit empty columns in the off-diagonal matrix B to limit possible nonzeros.
    # We assume C^-1 is completely dense.
    I, J, V = SparseArrays.findnz(A)
    # Columns of B with any entries. These are possible nonzeros of (B^T C^-1 B)
    B_nzcols = filter(i -> B.colptr[i] < B.colptr[i+1], 1:length(R))
    I_BTCB = IntType[i for i in B_nzcols for j in B_nzcols if i >= j]
    J_BTCB = IntType[j for i in B_nzcols for j in B_nzcols if i >= j]
    V_BTCB = FloatType[0.0 for _ in I_BTCB]
    append!(I, I_BTCB)
    append!(J, J_BTCB)
    append!(V, V_BTCB)
    reduced_matrix = SparseArrays.sparse(I, J, V)

    pivot_matrix = csc[P, P]
    # TODO: Use an option struct for PivotSolver and avoid branching here.
    if pivot_index_partition === nothing
        pivot_solver = PivotSolver(pivot_matrix; logger)
    else
        # TODO: What processing is necessary before using pivot_index_partition?
        pivot_solver = PivotSolver(pivot_matrix; blocks = pivot_index_partition, logger)
    end

    # This is some experimental code for getting the reduced matrix's nonzeros
    # experimentally, from a factorization.
    # For some reason, it didn't work. I forget why.
    #MadNLP.factorize!(pivot_solver)
    # Assuming that this matrix gives us a superset of all possible nonzeros
    #reduced_matrix = _sparse_schur(csc, pivot_indices)

    # The following unused code is for extracting the pivot submatrix explicitly
    # (e.g., not using csc[P, P]). I forget why I though this would be necessary...
    #remap = zeros(csc.n)
    #for (i, idx) in enumerate(pivot_indices)
    #    remap[idx] = i
    #end
    #colptr = IntType[1]
    #rowval = IntType[]
    #nzval = FloatType[]
    #for j in 1:csc.n # Columns
    #    # This just compresses the columns. It doesn't permute them if that is necessary
    #    if j in pivot_index_set
    #        pivot_nzs = filter(k -> csc.rowval[k] in pivot_index_set, csc.colptr[j]:(csc.colptr[j+1]-1))
    #        append!(rowval, remap[csc.rowval[pivot_nzs]])
    #        append!(nzval, csc.nzval[pivot_nzs])
    #        push!(colptr, colptr[end]+length(pivot_nzs))
    #    end
    #end
    #@assert !any(rowval .== 0)
    #
    #nnz = length(csc.nzval)
    #pivot_nzs = filter(i->(row[i] in pivot_index_set && col[i] in pivot_index_set), 1:nnz)
    ## Filter nonzeros to only contain the pivot submatrix
    #row = row[pivot_nzs]
    #col = col[pivot_nzs]
    #val = val[pivot_nzs]
    #row = remap[row]
    #col = remap[col]
    # This doesn't necessarily guarantee the order of nzvals either...
    #pivot_matrix = SparseArrays.sparse(row, col, val)
    #pivot_matrix = csc[pivot_indices, pivot_indices]
    #
    # Need to construct CSC explicitly so nonzeros don't get permuted or combined
    #pivot_matrix = SparseArrays.SparseMatrixCSC(
    #    pivot_dim,
    #    pivot_dim,
    #    colptr,
    #    rowval,
    #    nzval,
    #)
    I, J, V = SparseArrays.findnz(pivot_matrix)
    @assert all(I .>= J)
    # TODO: Allow passing options to subsolver
    reduced_solver = ReducedSolver(reduced_matrix; logger)

    timer.initialize += time() - t_start

    return SchurComplementSolver{FloatType,IntType}(
        csc, reduced_solver, pivot_solver, pivot_indices, timer
    )
end

function MadNLP.introduce(solver::SchurComplementSolver)
    rsolvername = MadNLP.introduce(solver.reduced_solver)
    ssolvername = MadNLP.introduce(solver.pivot_solver)
    pivot_dim = length(solver.pivot_indices)
    return (
        "A Schur-complement solver with reduced subsolver $(rsolvername) and Schur subsolver $(ssolvername)"
        * " operating on a pivot of size $(pivot_dim)x$(pivot_dim)"
    )
end

function MadNLP.factorize!(solver::SchurComplementSolver)
    # Assume csc.nzval has changed since the last call.
    # - Pivot matrix should be constructed with a view into csc.nzval, and therefore
    #   shouldn't need any manual update
    # - Reduced matrix will need to be recomputed, using the factorization of the
    #   pivot matrix
    # - Once the reduced matrix is computed, it needs to be loaded into the solver
    #   that uses it.
    # - To compute the reduced matrix, we will extract coordinates directly from
    #   the original matrix, csc, which have been updated.

    # Update nonzero values in the pivot solver
    #pivot_index_set = Set(solver.pivot_indices)
    #colptr = solver.csc.colptr
    #col = [j for j in 1:solver.csc.m for _ in colptr[j]:(colptr[j+1]-1)]
    #@assert length(col) == length(solver.csc.rowval)
    #nnz = length(solver.csc.nzval)
    #pivot_nzs = filter(
    #    i->(solver.csc.rowval[i] in pivot_index_set && col[i] in pivot_index_set),
    #    1:nnz,
    #)

    t_start = time()
    solver.pivot_solver.csc.nzval[:] = solver.csc[solver.pivot_indices, solver.pivot_indices].nzval
    solver.timer.factorize.update_pivot += time() - t_start

    t_pivot_start = time()
    MadNLP.factorize!(solver.pivot_solver)
    solver.timer.factorize.pivot += time() - t_pivot_start

    # KKT matrix has the following structure:
    #
    #      R  P
    #      -----
    # R) | A B^T |
    # P) | B  C  |
    #
    # The Schur complement WRT C is (A - B^T C^-1 B)

    # Get indices
    dim = solver.csc.n
    pivot_dim = length(solver.pivot_indices)
    reduced_dim = dim - pivot_dim
    pivot_index_set = Set(solver.pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)

    P = solver.pivot_indices
    R = reduced_indices

    A = solver.csc[R, R]
    # We need the off-diagonal block in the *permuted* matrix.
    # For entries with p<r in the original matrix, we must transpose the indices.
    B = solver.csc[P, R] + solver.csc[R, P]'

    #display(solver.csc[P, R])
    #display(solver.csc[R, P])
    #display(B)

    B_dense = Matrix(B)
    sol = copy(B_dense)
    t_solve_start = time()
    # Iterate over non-empty columns of B
    nonempty_cols = filter(j -> B.colptr[j] < B.colptr[j+1], 1:reduced_dim)
    compressed_sol = sol[:, nonempty_cols]
    # Backsolve over a matrix of RHSs. Note that this produces dense solutions
    # and relies on local extensions of `MadNLP.solve!`.
    MadNLP.solve!(solver.pivot_solver, compressed_sol)
    sol[:, nonempty_cols] = compressed_sol

    solver.timer.factorize.solve += time() - t_solve_start
    #println("B:")
    #display(B)
    #println("C:")
    #display(solver.pivot_solver.csc)
    t_multiply_start = time()
    term2 = B' * SparseArrays.sparse(sol)
    solver.timer.factorize.multiply += time() - t_multiply_start
    #println("B' C^-1 B")
    #display(term2)
    # The nonzero storage pattern here is not consistent.
    schur_complement = A - LinearAlgebra.tril(term2)
    #println("Schur complement matrix:")
    #display(schur_complement)
    #tol = 1e-10
    #nsmall = count(x -> abs(x) <= tol, schur_complement.nzval)
    #println("$nsmall entries with values below $tol")
    schur_lookup = Dict((i, j) => v for (i, j, v) in zip(SparseArrays.findnz(schur_complement)...))

    #println("Computed Schur complement:")
    #display(schur_complement)
    #display(Matrix(schur_complement))
    #println("schur_lookup:")
    #display(schur_lookup)

    t_update_start = time()
    for j in 1:reduced_dim # Iterate over columns
        # And over rows appearing in this column
        for k in solver.reduced_solver.csc.colptr[j]:(solver.reduced_solver.csc.colptr[j+1]-1)
            i = solver.reduced_solver.csc.rowval[k]
            solver.reduced_solver.csc.nzval[k] = get(schur_lookup, (i, j), 0.0)
        end
    end
    solver.timer.factorize.update_reduced += time() - t_update_start
    #println("Reduced solver's matrix before factorization:")
    #display(solver.reduced_solver.csc)
    t_reduced_start = time()
    MadNLP.factorize!(solver.reduced_solver)
    solver.timer.factorize.reduced += time() - t_reduced_start
    # Potentially, I could pre-compute some matrices that I need for the solve
    # of the Schur systems.
    solver.timer.factorize.total += time() - t_start
    return solver
end

function MadNLP.is_inertia(solver::SchurComplementSolver)
    # We already know the inertia for the Schur system
    return MadNLP.is_inertia(solver.reduced_solver)
    #return MadNLP.is_inertia(solver.reduced_solver) && MadNLP.is_inertia(solver.pivot_solver)
end

function MadNLP.inertia(solver::SchurComplementSolver)
    reduced_inertia = MadNLP.inertia(solver.reduced_solver)
    pos, zero, neg = reduced_inertia
    # We should be able to prove that the Schur complement system always has inertia
    # (dim, 0, dim). This is because it is decomposable.
    @assert length(solver.pivot_indices) % 2 == 0
    pivot_dim = length(solver.pivot_indices)
    return (pos + Int(pivot_dim/2), zero, neg + Int(pivot_dim/2))

    # Or we compute inertia by the Haynsworth formula
    #pivot_inertia = MadNLP.inertia(solver.pivot_solver)
    #println("Pivot system: (pos, zero, neg) = $pivot_inertia")
    #return reduced_inertia .+ pivot_inertia
end

function MadNLP.solve!(solver::SchurComplementSolver{T,INT}, rhs::Vector{T}) where {T,INT}
    t_start = time()
    # Partition rhs according to Schur complement coords
    dim = solver.csc.n
    pivot_dim = length(solver.pivot_indices)
    index_set = Set(solver.pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:dim)
    orig_rhs_reduced = rhs[reduced_indices]
    orig_rhs_pivot = rhs[solver.pivot_indices]

    #display(orig_rhs_reduced)
    # Original system:
    #      R  P
    #      -----
    # R) | A B^T | (x) = (r_x)
    # P) | B  C  | (y) = (r_y)
    #
    # Bx + Cy = r_y => y = C^-1 ( r_y - Bx )
    # Ax + B^Ty = r_x
    # => Ax + B^T C^-1 ( r_y - Bx ) = r_x
    # => ( A - B^T C^-1 B ) x = r_x - B^T C^-1 r_y

    # TODO: Cache these submatrices?
    P = solver.pivot_indices
    R = reduced_indices
    A = solver.csc[R, R]
    B = solver.csc[P, R] + solver.csc[R, P]'

    temp = copy(orig_rhs_pivot)
    MadNLP.solve!(solver.pivot_solver, temp)
    #println("C^-1 r_x")
    #display(temp)
    rhs_reduced = orig_rhs_reduced - B' * temp

    MadNLP.solve!(solver.reduced_solver, rhs_reduced)
    rhs_pivot = orig_rhs_pivot - B * rhs_reduced
    MadNLP.solve!(solver.pivot_solver, rhs_pivot)
    rhs[R] .= rhs_reduced
    rhs[P] .= rhs_pivot
    solver.timer.solve += time() - t_start
    return rhs
end

function MadNLP.improve!(solver::SchurComplementSolver{T,INT}) where {T,INT}
    return false
end

MadNLP.input_type(::Type{SchurComplementSolver}) = :csc
MadNLP.default_options(::Type{SchurComplementSolver}) = SchurComplementOptions()
# How to branch on subsolver types here?
# Parameterize the SchurComplementSolver type by its subsolver types?
MadNLP.is_supported(::Type{SchurComplementSolver}, ::Type{T}) where T <: AbstractFloat = true

function fill_upper_triangle(csc::SparseArrays.SparseMatrixCSC)
    I, J, V = SparseArrays.findnz(csc)
    strict_lower = filter(k -> I[k] > J[k], 1:length(I))
    upper_I = J[strict_lower]
    upper_J = I[strict_lower]
    upper_V = V[strict_lower]
    append!(I, upper_I)
    append!(J, upper_J)
    append!(V, upper_V)
    new = SparseArrays.sparse(I, J, V)
    return new
end

function remove_diagonal_nonzeros(
    csc::SparseArrays.SparseMatrixCSC,
    indices::Vector{Int},
)
    indexset = Set(indices)
    nnz = SparseArrays.nnz(csc)
    I, J, V = SparseArrays.findnz(csc)
    # Keep the entry if its row isn't in the index set or it's not on the diagonal
    to_retain = filter(k -> !(I[k] in indexset) || I[k] != J[k], 1:nnz)
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    newcsc = SparseArrays.sparse(I, J, V)
    return newcsc
end
