import SparseArrays
import MadNLP
import MadNLPHSL

# This seems not to work with a parameterized type?
#@kwdef mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions where {INT}
mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions
    ReducedSolver::Type
    SchurSolver::Type
    indices::Vector{Tuple{INT,INT}}
    #function SchurComplementOption(;
    #    ReducedSolver::Type = MadNLPHSL.Ma27Solver,
    #    SchurSolver::Type = MadNLPHSL.Ma27Solver,
    #    indices::Vector{Tuple{INT,INT}} = Tuple{Int32,Int32}[],
    #) where {INT}
    #    int = eltype(eltype(indices))
    #    return new{int}(ReducedSolver, SchurSolver, indices)
    #end
    SchurComplementOptions(;
        #MadNLPHSL.Ma27Solver,
        #MadNLPHSL.Ma27Solver,
        indices = Tuple{Int32,Int32}[],
    ) = new{eltype(eltype(indices))}(
        MadNLPHSL.Ma27Solver,
        MadNLPHSL.Ma27Solver,
        #Tuple{Int32,Int32}[],
        indices,
    )
end

struct SchurComplementSolver{T,INT} <: MadNLP.AbstractLinearSolver{T}
    csc::SparseArrays.SparseMatrixCSC{T,INT}
    reduced_solver::MadNLP.AbstractLinearSolver{T}
    schur_solver::MadNLP.AbstractLinearSolver{T}
    #ReducedSolver::Type
    #SchurSolver::Type
    # [(row, col),...]
    indices::Vector{Tuple{INT,INT}}
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
    #indices::Vector{Tuple{INT,INT}} = get(opt, "indices", Tuple{INT,INT}[]),
    ReducedSolver::Type = opt.ReducedSolver,
    SchurSolver::Type = opt.SchurSolver,
    indices::Vector{Tuple{INT,INT}} = opt.indices,
) where {T,INT}
    # TODO: Perform a complicated matrix reduction that reduces to a no-op when
    # indices arrays are empty.
    # TODO: I need to make sure these submatrices get updated when CSC changes
    # To do this, I will need to construct the submatrices directly from the
    # nz array of the original matrix.
    @assert csc.n == csc.m
    dim = csc.n
    schur_dim = length(indices)
    row_indices = [i for (i, j) in indices]
    col_indices = [j for (i, j) in indices]
    sym_indices = vcat(col_indices, row_indices)
    # We really only have to extract the LT portion of these indices, but
    # we don't change anything by extracting the zero entries due to the
    # UT portion.
    sym_index_set = Set(sym_indices)
    # Indices to preserve in the reduced-space system
    # TODO: Cache these indices?
    reduced_indices = filter(i -> !(i in sym_index_set), 1:dim)
    # NOTE: This is not the correct reduced system
    # TODO: Implement the correct reduced system
    reduced_matrix = csc[reduced_indices, reduced_indices]
    schur_matrix = csc[sym_indices, sym_indices]

    # TODO: Allow passing options to subsolver
    reduced_solver = ReducedSolver(reduced_matrix; logger)
    # TODO: This fails with an empty matrix. I'll set it to a dummy value as well.
    #schur_solver = SchurSolver(schur_matrix; logger)
    # At this point I don't even know the API that my Schur subsolver will use...
    # - I should try to leverage the exising API as much as possible
    # - I should probably force myself to use the existing API
    display(schur_matrix)
    schur_solver = SchurSolver(schur_matrix)

    # To mirror the behavior of "vanilla" solvers, we instantiate our sub-solvers
    # in this constructor.

    # TODO: Do I have to defer the instantiation of these linear solvers?
    # - I would like to do quite a bit of work on construction of this solver
    # - However, I will need the ability recompute factors when values are updated
    # - factorize! happens in-place after, implicitly, csc.nzval has been updated.
    floattype = eltype(csc.nzval)
    inttype = eltype(csc.rowval)
    return SchurComplementSolver{floattype,inttype}(
        csc, reduced_solver, schur_solver, indices
    )
end

function MadNLP.introduce(solver::SchurComplementSolver)
    rsolvername = MadNLP.introduce(solver.reduced_solver)
    ssolvername = MadNLP.introduce(solver.schur_solver)
    return "A Schur-complement solver with reduced subsolver $(rsolvername) and Schur subsolver $(ssolvername)"
end

function MadNLP.factorize!(solver::SchurComplementSolver)
    # csc.nzval has changed since the last call here. I need to:
    # - update values in the reduced solver; refactorize
    # - update the values in the schur complement system
    #   The Schur subsystem shouldn't really need a factorization, but depending on
    #   on what I actually use for the solver here, it may. E.g.:
    #   - With MA27, we'll need a factorization (with a custom pivot sequence, likely)
    #   - With something custom, I may need to factorize the 2x2 blocks?
    #   - Open question: Can I use cuDSS?

    # Note: To do this properly, I may have to construct the reduced/Schur submatrices
    # more explicitly to control the order of their nonzeros.
    #solver.schur_solver.csc.nzval .= solver.csc.nzval[schur_nz_indices]

    # We factorize the Schur system first because its factors are necessary to
    # construct the reduced system
    MadNLP.factorize!(solver.schur_solver)

    # KKT matrix has the following structure:
    #
    # | A B^T |
    # | B  C  |
    #
    # The Schur complement WRT C is (A - B^T C^-1 B)

    # Get indices
    dim = solver.csc.n
    schur_dim = length(solver.indices)
    reduced_dim = dim - schur_dim
    row_indices = [i for (i, j) in solver.indices]
    col_indices = [j for (i, j) in solver.indices]
    schur_indices = vcat(col_indices, row_indices)
    schur_index_set = Set(schur_indices)
    reduced_indices = filter(i -> !(i in schur_index_set), 1:dim)
    reduced_submatrix = solver.csc[reduced_indices, reduced_indices]

    # Combine the off-diagonal blocks. This is necessary
    off_diag = (
        solver.csc[schur_indices, reduced_indices]
        + solver.csc[reduced_indices, schur_indices]
    )

    offdiag_dense = Matrix(off_diag)
    sol = copy(offdiag_dense)
    for j in 1:reduced_dim
        MadNLP.solve!(solver.schur_solver, sol[:, j])
    end
    # This Schur complement needs to have a consistent nonzero pattern.
    # For now, it is "Full-sparse". If I start exploiting sparsity, I need to
    # be careful about this.
    schur_complement = reduced_submatrix - off_diag' * SparseArrays.sparse(sol)
    # Update nonzeros in the reduced system. The reduced system must contain
    # entries for all possible nonzeros we will need to update. I.e., we must have
    # constructed the reduced system previously when we initialized the linear solver
    solver.schur_solver.csc.nzval = schur_complement.nzval
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
    pos, zero, neg = MadNLP.inertia(solver.reduced_solver)
    # We should be able to prove that the Schur complement system always has inertia
    # (dim, 0, dim). This is because it is decomposable.
    schur_dim = length(solver.indices)
    return (pos + schur_dim, zero, neg + schur_dim)
    # Or we compute inertia by the Haynsworth formula
    # schur_inertia = MadNLP.inertia(solver.schur_solver)
    #return kkt_inertia .+ schur_inertia
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
