import SparseArrays
import MadNLP

mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions
#@kwdef mutable struct SchurComplementOptions{INT} <: MadNLP.AbstractOptions where {INT}
    ReducedSolver::Type
    SchurSolver::Type
    indices::Vector{Tuple{INT,INT}}
    SchurComplementOptions() = new{Int32}(MadNLPHSL.Ma27Solver, MadNLPHSL.Ma27Solver, [])
end

# TODO: Should this SparseSchurComplementSolver?
struct SchurComplementSolver{T,INT} <: MadNLP.AbstractLinearSolver{T}
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
    reduced_matrix = csc
    schur_dim = length(indices)
    row_indices = [i for (i, j) in indices]
    col_indices = [j for (i, j) in indices]
    schur_matrix = csc[row_indices, col_indices]

    # TODO: Allow passing options to subsolver
    reduced_solver = ReducedSolver(reduced_matrix; logger)
    # TODO: This fails with an empty matrix. I'll set it to a dummy value as well.
    #schur_solver = SchurSolver(schur_matrix; logger)
    schur_solver = reduced_solver

    # To mirror the behavior of "vanilla" solvers, we instantiate our sub-solvers
    # in this constructor.

    # TODO: Do I have to defer the instantiation of these linear solvers?
    # - I would like to do quite a bit of work on construction of this solver
    # - However, I will need the ability recompute factors when values are updated
    # - factorize! happens in-place after, implicitly, csc.nzval has been updated.
    floattype = eltype(csc.nzval)
    inttype = eltype(csc.rowval)
    return SchurComplementSolver{floattype,inttype}(
        reduced_solver, schur_solver, indices
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
    MadNLP.factorize!(solver.reduced_solver)
    return solver
end

function MadNLP.is_inertia(solver::SchurComplementSolver)
    return MadNLP.is_inertia(solver.reduced_solver) && MadNLP.is_inertia(solver.schur_solver)
end

function MadNLP.inertia(solver::SchurComplementSolver)
    kkt_inertia = MadNLP.inertia(solver.reduced_solver)
    schur_inertia = MadNLP.inertia(solver.schur_solver)
    # By the Haynsworth inertia additivity formula. I think this is right...
    #return kkt_inertia .+ schur_inertia
    return kkt_inertia
end

function MadNLP.solve!(solver::SchurComplementSolver{T,INT}, rhs::Vector{T}) where {T,INT}
    # Partition rhs according to Schur complement coords
    MadNLP.solve!(solver.reduced_solver, rhs)
    return rhs
end

MadNLP.input_type(::Type{SchurComplementSolver}) = :csc
MadNLP.default_options(::Type{SchurComplementSolver}) = SchurComplementOptions()
# How to branch on subsolver types here?
# Parameterize the SchurComplementSolver type by its subsolver types?
MadNLP.is_supported(::Type{SchurComplementSolver}, ::Type{T}) where T <: AbstractFloat = true
