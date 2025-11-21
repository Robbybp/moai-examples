import SparseArrays
import LinearAlgebra
import MadNLP
import MadNLPHSL
import HSL

using Printf

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

mutable struct SchurComplementBacksolveTimer
    total::Float64
    solve_schur::Float64
    solve_pivot::Float64
    compute_rhs::Float64
    SchurComplementBacksolveTimer() = new(0.0, 0.0, 0.0, 0.0)
end

mutable struct SchurComplementTimer
    initialize::Float64
    factorize::SchurComplementFactorizeTimer
    solve::Float64 # TODO: Replace this with SchurComplementBacksolveTimer everywhere
    solve_timer::SchurComplementBacksolveTimer
    SchurComplementTimer() = new(0.0, SchurComplementFactorizeTimer(), 0.0, SchurComplementBacksolveTimer())
end

function Base.show(io::IO, timer::SchurComplementTimer)
    println(io, "SchurComplementSolver timing information")
    println(io, "----------------------------------------")
    println(io, "initialize: $(timer.initialize)")
    println(io, "factorize:  $(timer.factorize.total)")
    println(io, "  factorize Schur:             $(timer.factorize.reduced)")
    println(io, "  factorize pivot:             $(timer.factorize.pivot)")
    #println(io, "  update_pivot:   $(timer.factorize.update_pivot)")
    println(io, "  construct Schur (backsolve): $(timer.factorize.solve)")
    println(io, "  construct Schur (multiply):  $(timer.factorize.multiply)")
    #println(io, "  update_reduced: $(timer.factorize.update_reduced)")
    #! format: off
    other = (
        timer.factorize.total
        - timer.factorize.reduced
        - timer.factorize.pivot
        #- timer.factorize.update_pivot
        - timer.factorize.solve
        - timer.factorize.multiply
        #- timer.factorize.update_reduced
    )
    #! format: on
    println(io, "  other:                       $(other)")
    println(io, "solve:      $(timer.solve)")
    println(io, "----------------------------------------")
    return
end

mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions
    ReducedSolver::Type
    PivotSolver::Type
    pivot_indices::Vector{INT}
    pivot_solver_opt::Union{Nothing,MadNLP.AbstractOptions}
    # TODO: Parameterize this constructor by types
    function SchurComplementOptions(;
        # TODO: Non-third party default subsolver
        ReducedSolver = MadNLPHSL.Ma27Solver,
        PivotSolver = MadNLPHSL.Ma27Solver,
        pivot_indices = Int32[],
        # NOTE: If pivot_solver_opt is not specified, we will use
        # default_options(PivotSolver) in the SchurComplementSolver constructor.
        pivot_solver_opt = nothing,
    )
        return new{eltype(pivot_indices)}(
            ReducedSolver,
            PivotSolver,
            pivot_indices,
            pivot_solver_opt,
        )
    end
end

# TODO: Parameterize this by the inner solver
struct SchurComplementSolver{T,INT} <: MadNLP.AbstractLinearSolver{T}
    csc::SparseArrays.SparseMatrixCSC{T,INT}
    reduced_solver::MadNLP.AbstractLinearSolver{T}
    # Specifying these a bit further didn't really help
    #reduced_solver::Union{MadNLPHSL.Ma27Solver{T,INT},MadNLPHSL.Ma57Solver{T,INT}}
    pivot_solver::MadNLP.AbstractLinearSolver{T}
    #pivot_solver::Union{MadNLPHSL.Ma27Solver{T,INT},MadNLPHSL.Ma57Solver{T,INT},BlockTriangularSolver}

    # These are indices on which we pivot rows and columns.
    pivot_indices::Vector{INT}
    timer::SchurComplementTimer

    # Intermediate data structures required for in-place operations
    pivot_nz::Vector{INT}
    B_nzcols::Vector{INT}
    B_compressed::SparseArrays.SparseMatrixCSC{T,INT}
    compressed_intermediate_sol::Matrix{T}
    #schur_lookup::Dict{Tuple{INT,INT},T}
    Anz_remap::Vector{INT}
    BTCBnz_remap::Vector{INT}
end

# These help with type stability in the factorize! method, but it's not clear
# they improve performance overall...
function get_matrix(solver::SchurComplementSolver{T,INT})::SparseArrays.SparseMatrixCSC{T,INT} where {T,INT}
    return solver.csc
end
function get_pivot_solver_matrix(solver::SchurComplementSolver{T,INT})::SparseArrays.SparseMatrixCSC{T,INT} where {T,INT}
    return solver.pivot_solver.csc
end
function get_reduced_solver_matrix(solver::SchurComplementSolver{T,INT})::SparseArrays.SparseMatrixCSC{T,INT} where {T,INT}
    return solver.reduced_solver.csc
end
function factorize_pivot!(solver::SchurComplementSolver{T,INT}) where {T,INT}
    return MadNLP.factorize!(solver.pivot_solver)
end
function factorize_reduced!(solver::SchurComplementSolver{T,INT}) where {T,INT}
    return MadNLP.factorize!(solver.reduced_solver)
end
function solve_pivot!(solver::SchurComplementSolver{T,INT}, sol::Matrix{T}) where {T,INT}
    return MadNLP.solve!(solver.pivot_solver, sol)
end
# This actually gives my more unspecificed types in code_warntype...
#function get_matrix(solver::MadNLPHSL.Ma27Solver{T,INT})::SparseArrays.SparseMatrixCSC{T,INT} where {T,INT}
#    return solver.csc
#end
#function get_matrix(solver::MadNLPHSL.Ma57Solver{T,INT})::SparseArrays.SparseMatrixCSC{T,INT} where {T,INT}
#    return solver.csc
#end
function get_matrix(solver::MadNLP.AbstractLinearSolver{T})::SparseArrays.SparseMatrixCSC{T,Int32} where {T}
    return solver.csc
end

function SchurComplementSolver(
    csc::SparseArrays.SparseMatrixCSC{T,INT};
    opt::SchurComplementOptions = SchurComplementOptions(),
    logger::MadNLP.MadNLPLogger = MadNLP.MadNLPLogger(),
) where {T,INT}
    ReducedSolver = opt.ReducedSolver
    PivotSolver = opt.PivotSolver
    pivot_indices = opt.pivot_indices
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
    @assert all(Iorig .== csc.rowval)
    # Assert that only lower triangular entries are provided.
    # If parts of the upper triangle are provided, our Schur complement below
    # will be incorrect.
    @assert all(Iorig .>= Jorig)
    # Make sure the original matrix contains no duplicates.
    # I forget why this is necessary, but I think it has something to do with
    # extracting indices to update the reduced and pivot matrices.
    # COO has no duplicates:
    @assert length(Set(zip(Iorig, Jorig))) == length(Iorig)
    # CSC has same number of duplicates as COO:
    @assert length(Vorig) == length(csc.nzval)

    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)
    R = reduced_indices
    P = pivot_indices

    nnz = SparseArrays.nnz(csc)
    I_is_pivot_nz = Iorig .∈ (pivot_index_set,)
    J_is_pivot_nz = Jorig .∈ (pivot_index_set,)
    # By convention, the second half of pivot indices correspond to constraints.
    pivot_con_indices = Set(pivot_indices[Int(pivot_dim / 2 + 1):pivot_dim])
    is_pivot_con_diagonal = (Iorig .== Jorig) .& (Jorig .∈ (pivot_con_indices,))
    pivot_nz = findall(I_is_pivot_nz .& J_is_pivot_nz .& .!is_pivot_con_diagonal)
    reduced_nz = findall(.!I_is_pivot_nz .& .!J_is_pivot_nz)
    offdiag_nz = findall(I_is_pivot_nz .⊻ J_is_pivot_nz)
    # This no longer holds because I filter out constraint regularization nonzeros.
    #@assert length(pivot_nz) + length(reduced_nz) + length(offdiag_nz) == nnz

    # I have the indices of the nzval for each submatrix.
    pivot_remap = zeros(IntType, dim)
    # Remap pivot indices to start from 1
    pivot_remap[pivot_indices] .= 1:pivot_dim
    Ipivot = pivot_remap[Iorig[pivot_nz]]
    Jpivot = pivot_remap[Jorig[pivot_nz]]
    Vpivot = Vorig[pivot_nz]
    # `sparse` appears to sort by col (mandatory for CSC) then by row-within-col.
    # I need to reproduce this sorting.
    pivot_bycol = sortperm(pivot_dim .* Jpivot .+ Ipivot)
    Ipivot = Ipivot[pivot_bycol]
    Jpivot = Jpivot[pivot_bycol]
    Vpivot = Vpivot[pivot_bycol]
    pivot_nz = pivot_nz[pivot_bycol]
    pivot_matrix = SparseArrays.sparse(Ipivot, Jpivot, Vpivot, pivot_dim, pivot_dim)
    # If I don't sort nonzeros by column, I expect this to fail
    @assert all(Ipivot .== pivot_matrix.rowval)
    @assert all(Ipivot .>= Jpivot)

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
    # Here, we exploit empty columns in the off-diagonal matrix B to limit possible nonzeros.
    # We assume C^-1 is completely dense.
    _t = time()
    AI, AJ, AV = SparseArrays.findnz(A)
    A_nnz = SparseArrays.nnz(A)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] findnz")
    # Columns of B with any entries. These are possible nonzeros of (B^T C^-1 B)
    B_nzcols = filter(i -> B.colptr[i] < B.colptr[i+1], 1:length(R))
    Bcolset = Set(B_nzcols)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Bcols")
    AI_in_Bcols = AI .∈ (Bcolset,)
    AJ_in_Bcols = AJ .∈ (Bcolset,)
    Anz_in_Bcols_indicator = AI_in_Bcols .& AJ_in_Bcols

    # These are the indices of nonzeros that are or aren't in (B^T C^1 B)
    Anz_in_Bcols = findall(Anz_in_Bcols_indicator)
    Anz_notin_Bcols = findall(.!Anz_in_Bcols_indicator)
    AI_notin_Bcols = AI[Anz_notin_Bcols]
    AJ_notin_Bcols = AJ[Anz_notin_Bcols]
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Bcol indices")

    # I was experimenting with changing the ordering here...
    #I_BTCB = IntType[i for j in B_nzcols for i in B_nzcols if i >= j]
    #J_BTCB = IntType[j for j in B_nzcols for i in B_nzcols if i >= j]
    I_BTCB = IntType[i for i in B_nzcols for j in B_nzcols if i >= j]
    J_BTCB = IntType[j for i in B_nzcols for j in B_nzcols if i >= j]
    V_BTCB = FloatType[0.0 for _ in I_BTCB]
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] BTCB")

    # S is the Schur complement:
    #   S = A - B^T C^-1 B
    # These are unique nonzeros coordinates and now must be sorted.
    BTCB_nnz = length(I_BTCB)
    SI = vcat(I_BTCB, AI_notin_Bcols)
    SJ = vcat(J_BTCB, AJ_notin_Bcols)
    S_nnz = BTCB_nnz + length(Anz_notin_Bcols)
    SV = zeros(FloatType, S_nnz)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Alloc-Schur")
    # Sort by column, then row
    S_nzperm = sortperm(reduced_dim .* SJ .+ SI)
    SI = SI[S_nzperm]
    SJ = SJ[S_nzperm]
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Sort")
    reduced_matrix = SparseArrays.sparse(SI, SJ, SV)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Construct Schur CSC")
    # If this fails, nonzeros have been reordered
    @assert all(reduced_matrix.rowval .== SI)

    # I need to use B_col_remap here to correctly calculate the nonzero indices
    B_col_remap = zeros(IntType, reduced_dim)
    B_col_remap[B_nzcols] .= 1:length(B_nzcols)

    # We will update the Schur complement (reduced) matrix with:
    # solver.reduced_solver.csc.nzval[solver.Anz_remap] .+= A.nzval
    # solver.reduced_solver.csc.nzval[solver.BTCBnz_remap] .-= BTCBnz

    Anz_remap = zeros(IntType, A_nnz)
    # Assign, to positions in the original nonzeros, positions in the combined nonzeros
    Anz_remap[Anz_notin_Bcols] .= BTCB_nnz .+ (1:length(Anz_notin_Bcols))
    Anz_remap[Anz_in_Bcols] .= (
        # This assumes nonzeros of (B^T C^-1 B) are sorted by column first
        #B_col_remap[AJ[Anz_in_Bcols]] .* (B_col_remap[AJ[Anz_in_Bcols]] .- 1) ./ 2
        #.+ B_col_remap[AI[Anz_in_Bcols]]
        #
        # This assumes nonzeros of (B^T C^-1 B) are sorted by row first
        B_col_remap[AI[Anz_in_Bcols]] .* (B_col_remap[AI[Anz_in_Bcols]] .- 1) ./ 2
        .+ B_col_remap[AJ[Anz_in_Bcols]]
    )
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] original remaps")
    @assert !any(Anz_remap .== 0)
    S_nzperm_old_to_new = invperm(S_nzperm)
    Anz_remap = S_nzperm_old_to_new[Anz_remap]
    BTCBnz_remap = S_nzperm_old_to_new[1:BTCB_nnz]
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] permuted remaps")

    # Allocate memory for intermediate solution C^-1 B
    IB, JB, VB = SparseArrays.findnz(B)
    JB = B_col_remap[JB]
    B_compressed = SparseArrays.sparse(IB, JB, VB, pivot_dim, length(B_nzcols))
    @assert all(B_compressed.rowval .== IB)
    compressed_intermediate_sol = zeros(FloatType, pivot_dim, length(B_nzcols))

    # NOTE: We defer evaluation of default_options until this point because MadNLP
    # assumes no dependency among options. This way we call default_options only after
    # we know what PivotSolver was specified as (instead of just calling it on the default
    # PivotSolver (MA27) earlier).
    pivot_solver_opt = something(opt.pivot_solver_opt, MadNLP.default_options(PivotSolver))
    pivot_solver = PivotSolver(pivot_matrix; opt = pivot_solver_opt, logger)

    # TODO: Allow passing options to reduced solver
    reduced_solver = ReducedSolver(reduced_matrix; logger)

    #println("Time spend initializing: $(time() - t_start)")
    timer.initialize += time() - t_start

    return SchurComplementSolver{FloatType,IntType}(
        csc,
        reduced_solver,
        pivot_solver,
        pivot_indices,
        timer,
        pivot_nz,
        B_nzcols,
        B_compressed,
        compressed_intermediate_sol,
        #schur_lookup,
        Anz_remap,
        BTCBnz_remap,
    )
end

function MadNLP.introduce(solver::SchurComplementSolver)
    rsolvername = MadNLP.introduce(solver.reduced_solver)
    psolvername = MadNLP.introduce(solver.pivot_solver)
    pivot_dim = length(solver.pivot_indices)
    return (
        "A Schur-complement solver with reduced-space subsolver $(rsolvername) and pivot subsolver $(psolvername)"
        * " operating on a pivot of size $(pivot_dim)x$(pivot_dim)"
    )
end

function MadNLP.factorize!(solver::SchurComplementSolver)
    # Assume csc.nzval has changed since the last call.
    # - Update pivot matrix directly from original matrix's nzval
    # - Reduced matrix will need to be recomputed, using the factorization of the
    #   pivot matrix
    # - Once the reduced matrix is computed, it needs to be loaded into the solver
    #   that uses it.

    # Update nonzero values in the pivot solver
    t_start = time()
    csc = get_matrix(solver)
    #pivot_csc = get_matrix(solver.pivot_solver)
    #reduced_csc = get_matrix(solver.reduced_solver)
    pivot_csc = get_pivot_solver_matrix(solver)
    reduced_csc = get_reduced_solver_matrix(solver)

    dim = csc.m
    pivot_dim = length(solver.pivot_indices)
    pivot_index_set = Set(solver.pivot_indices)
    pivot_csc.nzval .= csc.nzval[solver.pivot_nz]
    solver.timer.factorize.update_pivot += time() - t_start

    t_pivot_start = time()
    #println("FACTORIZING PIVOT MATRIX")
    #MadNLP.factorize!(solver.pivot_solver)
    factorize_pivot!(solver)
    solver.timer.factorize.pivot += time() - t_pivot_start

    # KKT matrix has the following structure:
    #
    #      R  P
    #      -----
    # R) | A B^T |
    # P) | B  C  |
    #
    # The Schur complement WRT C is (A - B^T C^-1 B)
    # C is the "pivot matrix"

    # Get indices
    reduced_dim = dim - pivot_dim
    # TODO: Cache reduced indices as well?
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:dim)

    P = solver.pivot_indices
    R = reduced_indices

    A = csc[R, R]
    # We need the off-diagonal block in the *permuted* matrix.
    # We are given the lower triangle of the *original* matrix.
    # For entries with p<r in the original matrix, we must transpose the indices.
    B = csc[P, R] + csc[R, P]'

    # TODO: Cache A and B and update in-place
    # I'll have to figure out how to handle B's indices...
    #solver.A.nzval .= csc.nzval[solver.reduced_nz]
    #solver.B.nzval .= csc.nzval[solver.offdiag_nz]

    t_solve_start = time()
    # Iterate over non-empty columns of B
    nonempty_cols = solver.B_nzcols
    solver.B_compressed.nzval .= B.nzval
    B_compressed = solver.B_compressed

    # This stores the solution to C^-1 B. The memory is pre-allocated.
    solver.compressed_intermediate_sol .= 0.0
    # I could potentially vectorize this loop
    for j in 1:length(nonempty_cols)
        for k in B_compressed.colptr[j]:(B_compressed.colptr[j+1]-1)
            i = B_compressed.rowval[k]
            solver.compressed_intermediate_sol[i, j] += B_compressed.nzval[k]
        end
    end
    compressed_sol = solver.compressed_intermediate_sol

    # Backsolve over a matrix of RHSs. Note that this produces dense solutions
    # and relies on local extensions of `MadNLP.solve!`.
    #println("BACKSOLVING TO CONSTRUCT SCHUR COMPLEMENT")
    #MadNLP.solve!(solver.pivot_solver, compressed_sol)
    solve_pivot!(solver, compressed_sol)

    solver.timer.factorize.solve += time() - t_solve_start
    t_multiply_start = time()
    # While B_compressed is sparse, it may be better to convert to dense here?
    # ^ Not in any benchmark I have done.
    #compressed_term2 = Matrix(B_compressed') * compressed_sol
    compressed_term2 = B_compressed' * compressed_sol
    # Make sure I get a dense matrix back from this multiplication.
    @assert compressed_term2 isa Matrix
    solver.timer.factorize.multiply += time() - t_multiply_start

    # I need tril(term2) to have a consistent sparsity structure.
    # How hard would it be to have tril(term2) already allocated?
    #
    # For now, I'll store the nonzeros, sorted by (row, col). Then I'll apply
    # the cached permutation.
    # I can work on doing this in-place later.
    #
    # NOTE: It is critical that the sorting here (e.g., row-then-col) matches
    # the cached BTCBnz_remap permutation.
    #I_BTCBnz = [i for j in nonempty_cols for i in nonempty_cols if i >= j]
    #J_BTCBnz = [j for j in nonempty_cols for i in nonempty_cols if i >= j]
    I_BTCBnz = [i for i in nonempty_cols for j in nonempty_cols if i >= j]
    J_BTCBnz = [j for i in nonempty_cols for j in nonempty_cols if i >= j]
    BTCBnz = [compressed_term2[ipos, jpos] for (ipos, i) in enumerate(nonempty_cols) for (jpos, j) in enumerate(nonempty_cols) if i >= j]
    #BTCB = SparseArrays.sparse(I_BTCBnz, J_BTCBnz, BTCBnz)

    # In-place update without hashing
    t_update_start = time()
    reduced_csc.nzval .= 0.0
    #display(solver.reduced_solver.csc)
    reduced_csc.nzval[solver.Anz_remap] .+= A.nzval
    reduced_csc.nzval[solver.BTCBnz_remap] .-= BTCBnz

    #display(A)
    #display(BTCB)
    #display(reduced_csc)

    solver.timer.factorize.update_reduced += time() - t_update_start
    #println("Reduced solver's matrix before factorization:")
    #display(reduced_csc)
    t_reduced_start = time()
    #println("FACTORIZING SCHUR COMPLEMENT MATRIX")
    #MadNLP.factorize!(solver.reduced_solver)
    factorize_reduced!(solver)
    solver.timer.factorize.reduced += time() - t_reduced_start
    # Potentially, I could pre-compute some matrices that I need for the solve
    # of the Schur systems.
    solver.timer.factorize.total += time() - t_start
    #println("Time spend factorizing: $(time() - t_start)")
    return solver
end

function MadNLP.is_inertia(solver::SchurComplementSolver)
    # We already know the inertia for the Schur system
    return MadNLP.is_inertia(solver.reduced_solver)
    # If we use the Haynsworth formula:
    #return MadNLP.is_inertia(solver.reduced_solver) && MadNLP.is_inertia(solver.pivot_solver)
end

function MadNLP.inertia(solver::SchurComplementSolver)
    reduced_inertia = MadNLP.inertia(solver.reduced_solver)
    pos, zero, neg = reduced_inertia
    # We should be able to prove that the Schur complement system always has inertia
    # (dim, 0, dim).
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

    t_solve_pivot = 0.0
    t_compute_rhs = 0.0
    t_solve_schur = 0.0

    temp = copy(orig_rhs_pivot)
    _t = time()
    MadNLP.solve!(solver.pivot_solver, temp)
    t_solve_pivot += time() - _t

    _t = time()
    rhs_reduced = orig_rhs_reduced - B' * temp
    t_compute_rhs += time() - _t

    _t = time()
    MadNLP.solve!(solver.reduced_solver, rhs_reduced)
    t_solve_schur += time() - _t

    _t = time()
    rhs_pivot = orig_rhs_pivot - B * rhs_reduced
    t_compute_rhs += time() - _t

    _t = time()
    MadNLP.solve!(solver.pivot_solver, rhs_pivot)
    t_solve_pivot += time() - _t

    rhs[R] .= rhs_reduced
    rhs[P] .= rhs_pivot
    t_total = time() - t_start
    solver.timer.solve += t_total
    solver.timer.solve_timer.total += t_total
    solver.timer.solve_timer.solve_schur += t_solve_schur
    solver.timer.solve_timer.solve_pivot += t_solve_pivot
    solver.timer.solve_timer.compute_rhs += t_compute_rhs
    #println("Time spend in backsolve: $(time() - t_start)")
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

# This is used elsewhere, basically just for inspecting the full, symmetric matrix.
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

# This was really slow compared to doing basically the same thing inline above,
# for some reason...
function remove_diagonal_nonzeros(
    csc::SparseArrays.SparseMatrixCSC,
    indices::Vector{Int},
)
    _t = time()
    indexset = Set(indices)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] indexset")
    nnz = SparseArrays.nnz(csc)
    I, J, V = SparseArrays.findnz(csc)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] findnz")
    # Keep the entry if its row isn't in the index set or it's not on the diagonal
    to_retain = filter(k -> I[k] != J[k] || !(I[k] in indexset), 1:nnz)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] filter")
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] extract indices")
    newcsc = SparseArrays.sparse(I, J, V, csc.m, csc.n)
    #dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] sparse")
    return newcsc
end

function refine!(
    sol::Vector{Float64},
    solver::MadNLP.AbstractLinearSolver,
    rhs::Vector{Float64};
    tol::Float64 = 1e-8,
    max_iter::Int = 10,
    full_matrix::Union{Nothing,SparseArrays.SparseMatrixCSC} = nothing,
    tril_to_full_view::Union{Nothing,SubArray} = nothing,
)
    #println("Starting iterative refinement")
    _t = time()
    if full_matrix === nothing
        matrix = fill_upper_triangle(solver.csc)
    else
        @assert tril_to_full_view !== nothing
        # This could be done outside of this function, saving us an argument, but
        # I would like it to be include in the residual computation time
        full_matrix.nzval .= tril_to_full_view
        matrix = full_matrix
    end
    residual = rhs - matrix * sol
    resid_norm = LinearAlgebra.norm(residual, Inf)
    t_resid = time() - _t
    #println("[$(@sprintf("%1.2f", t_resid))] Compute residual")
    t_backsolve = 0.0

    iter_count = 0
    if resid_norm <= tol
        return (; success = true, iterations = iter_count, residual_norm = resid_norm, t_resid, t_backsolve)
    end
    for i in 1:max_iter
        correction = copy(residual)
        _t = time()
        MadNLP.solve!(solver, correction)
        dt = time() - _t;
        #println("[$(@sprintf("%1.2f", dt))] Backsolve")
        t_backsolve += dt
        sol .+= correction
        residual = rhs - matrix * sol
        _t = time()
        resid_norm = LinearAlgebra.norm(residual, Inf)
        dt = time() - _t
        #println("[$(@sprintf("%1.2f", dt))] Update solution and compute residual")
        t_resid += dt
        iter_count = i
        if resid_norm <= tol
            break
        end
    end
    return (;
        success = resid_norm <= tol,
        iterations = iter_count,
        residual_norm = resid_norm,
        t_resid,
        t_backsolve,
    )
end
