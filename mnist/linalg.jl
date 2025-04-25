import SparseArrays
import LinearAlgebra
import MadNLP
import MadNLPHSL

# This seems not to work with a parameterized type?
#@kwdef mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions where {INT}
mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions
    ReducedSolver::Type
    SchurSolver::Type
    # We're going to perform a symmetric reduction, so we really only need one set of indices
    pivot_indices::Vector{INT}
    #function SchurComplementOption(;
    #    ReducedSolver::Type = MadNLPHSL.Ma27Solver,
    #    SchurSolver::Type = MadNLPHSL.Ma27Solver,
    #    pivot_indices::Vector{INT} = Tuple{Int32,Int32}[],
    #) where {INT}
    #    int = eltype(eltype(pivot_indices))
    #    return new{int}(ReducedSolver, SchurSolver, pivot_indices)
    #end
    SchurComplementOptions(;
        #MadNLPHSL.Ma27Solver,
        #MadNLPHSL.Ma27Solver,
        pivot_indices = [],
    ) = new{eltype(pivot_indices)}(
        MadNLPHSL.Ma27Solver,
        MadNLPHSL.Ma27Solver,
        #Tuple{Int32,Int32}[],
        pivot_indices,
    )
end

struct SchurComplementSolver{T,INT} <: MadNLP.AbstractLinearSolver{T}
    csc::SparseArrays.SparseMatrixCSC{T,INT}
    reduced_solver::MadNLP.AbstractLinearSolver{T}
    # "SchurSolver" is a bit ambiguous. TODO: find a better name.
    schur_solver::MadNLP.AbstractLinearSolver{T}
    # These are indices on which we pivot rows and columns.
    pivot_indices::Vector{INT}
end

function SchurComplementSolver(
    csc::SparseArrays.SparseMatrixCSC{T,INT};
    # TODO: Make this a SchurComplementOptions struct
    #opt::Dict = Dict(),
    opt::SchurComplementOptions = SchurComplementOptions(),
    logger::MadNLP.MadNLPLogger = MadNLP.MadNLPLogger(),
    # TODO: non-third-party default
    #ReducedSolver::Type = get(opt, "reduced_solver", MadNLPHSL.Ma27Solver),
    #SchurSolver::Type = get(opt, "schur_solver", MadNLPHSL.Ma27Solver),
    #pivot_indices::Vector{INT} = get(opt, "pivot_indices", INT[]),
    ReducedSolver::Type = opt.ReducedSolver,
    SchurSolver::Type = opt.SchurSolver,
    pivot_indices::Vector{INT} = opt.pivot_indices,
) where {T,INT}
    FloatType = eltype(csc.nzval)
    IntType = eltype(csc.rowval)
    # TODO: I need to make sure these submatrices get updated when CSC changes
    # To do this, I will need to construct the submatrices directly from the
    # nz array of the original matrix.
    @assert csc.n == csc.m
    dim = csc.n
    pivot_dim = length(pivot_indices)
    reduced_dim = dim - pivot_dim
    I, J, V = SparseArrays.findnz(csc)
    # Assert that only lower triangular entries are provided.
    # If parts of the upper triangle are provided, our Schur complement below
    # will be incorrect.
    @assert all(I .>= J)

    # For now, the reduced matrix is dense. If we exploit sparsity, we will need
    # to consider the sparsity structure here.
    I = IntType[i for i in 1:reduced_dim for j in 1:reduced_dim]
    J = IntType[j for i in 1:reduced_dim for j in 1:reduced_dim]
    V = FloatType[0.0 for _ in I]
    reduced_matrix = SparseArrays.sparse(I, J, V)

    # The pivot indices correspond to variable and constraint indices that have
    # a nonsingular Jacobian. We pivot on the symmetric block:
    #
    # | W_yy  ∇_y g^T |
    # | ∇_y g         |
    #
    # Extracting these indices gives us the lower triangle of this matrix, which
    # is exactly what we want.
    pivot_matrix = csc[pivot_indices, pivot_indices]
    # TODO: Allow passing options to subsolver
    reduced_solver = ReducedSolver(reduced_matrix; logger)
    schur_solver = SchurSolver(pivot_matrix; logger)

    return SchurComplementSolver{FloatType,IntType}(
        csc, reduced_solver, schur_solver, pivot_indices
    )
end

function MadNLP.introduce(solver::SchurComplementSolver)
    rsolvername = MadNLP.introduce(solver.reduced_solver)
    ssolvername = MadNLP.introduce(solver.schur_solver)
    return "A Schur-complement solver with reduced subsolver $(rsolvername) and Schur subsolver $(ssolvername)"
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

    MadNLP.factorize!(solver.schur_solver)

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
    println("Reduced dim = $reduced_dim")
    for j in 1:reduced_dim
        MadNLP.solve!(solver.schur_solver, sol[:, j])
    end
    # The nonzero storage pattern here is not consistent.
    schur_complement = A - LinearAlgebra.tril(B' * SparseArrays.sparse(sol))
    schur_lookup = Dict((i, j) => v for (i, j, v) in zip(SparseArrays.findnz(schur_complement)...))

    #println("Computed Schur complement:")
    #display(schur_complement)
    #display(Matrix(schur_complement))
    #println("schur_lookup:")
    #display(schur_lookup)

    for j in 1:reduced_dim # Iterate over columns
        # And over rows appearing in this column
        for k in solver.reduced_solver.csc.colptr[j]:(solver.reduced_solver.csc.colptr[j+1]-1)
            i = solver.reduced_solver.csc.rowval[k]
            if (i, j) in keys(schur_lookup)
                solver.reduced_solver.csc.nzval[k] = schur_lookup[i,j]
            else
                solver.reduced_solver.csc.nzval[k] = 0.0
            end
        end
    end
    #println("Reduced solver's matrix before factorization:")
    #display(solver.reduced_solver.csc)
    #display(Matrix(solver.reduced_solver.csc))
    MadNLP.factorize!(solver.reduced_solver)
    # Potentially, I could pre-compute some matrices that I need for the solve
    # of the Schur systems.
    return solver
end

function MadNLP.is_inertia(solver::SchurComplementSolver)
    # We already know the inertia for the Schur system
    return MadNLP.is_inertia(solver.reduced_solver)
    #return MadNLP.is_inertia(solver.reduced_solver) && MadNLP.is_inertia(solver.schur_solver)
end

function MadNLP.inertia(solver::SchurComplementSolver)
    reduced_inertia = MadNLP.inertia(solver.reduced_solver)
    pos, zero, neg = reduced_inertia
    println("Reduced system: (pos, zero, neg) = ($pos, $zero, $neg)")
    # We should be able to prove that the Schur complement system always has inertia
    # (dim, 0, dim). This is because it is decomposable.
    @assert length(solver.pivot_indices) % 2 == 0
    pivot_dim = length(solver.pivot_indices)
    return (pos + Int(pivot_dim/2), zero, neg + Int(pivot_dim/2))

    # Or we compute inertia by the Haynsworth formula
    #pivot_inertia = MadNLP.inertia(solver.schur_solver)
    #println("Pivot system: (pos, zero, neg) = $pivot_inertia")
    #return reduced_inertia .+ pivot_inertia
end

function MadNLP.solve!(solver::SchurComplementSolver{T,INT}, rhs::Vector{T}) where {T,INT}
    # Partition rhs according to Schur complement coords
    dim = solver.csc.n
    row_indices = [i for (i, j) in solver.indices]
    col_indices = [j for (i, j) in solver.indices]
    sym_indices = vcat(col_indices, row_indices)
    sym_index_set = Set(sym_indices)
    reduced_indices = filter(i -> !(i in sym_index_set), 1:dim)
    rhs_reduced = rhs[reduced_indices]
    rhs_schur = rhs[sym_indices]
    # TODO: Compute the correct RHSs

    MadNLP.solve!(solver.reduced_solver, rhs_reduced)
    MadNLP.solve!(solver.schur_solver, rhs_schur)
    rhs[reduced_indices] .= rhs_reduced
    rhs[sym_indices] .= rhs_schur
    return rhs
end

MadNLP.input_type(::Type{SchurComplementSolver}) = :csc
MadNLP.default_options(::Type{SchurComplementSolver}) = SchurComplementOptions()
# How to branch on subsolver types here?
# Parameterize the SchurComplementSolver type by its subsolver types?
MadNLP.is_supported(::Type{SchurComplementSolver}, ::Type{T}) where T <: AbstractFloat = true
