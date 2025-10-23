using Printf

import JuMP
import MadNLP
import MadNLPHSL

include("../linalg.jl")
include("../btsolver.jl")
include("../formulation.jl")
include("../nlpmodels.jl")
include("../kkt-partition.jl")

function _get_ma57_data(ma57)
    return (;
        # This is number of entries in the factors (not number of bytes)
        factor_size = ma57.info[14],
        # These are flops in assembly and elimination processes.
        flops = ma57.rinfo[3] + ma57.rinfo[4],
        n2by2 = ma57.info[22],
        status_code = ma57.info[1],
    )
end

function _get_ma86_data(ma86)
    return (;
        factor_size = ma86.info.num_factor,
        flops = ma86.info.num_flops,
        n2by2 = ma86.info.num_two,
        status_code = ma86.info.flag,
    )
end

SOLVER_DATA_GETTER = Dict(
    MadNLPHSL.Ma57Solver => _get_ma57_data,
    MadNLPHSL.Ma86Solver => _get_ma86_data,
)
function _get_solver_data_getter(Solver)
    # Return an empty namedtuple if we don't have a custom getter...
    # This isn't right, because these NamedTuples need to have the same fields.
    return get(SOLVER_DATA_GETTER, Solver, x -> (; factor_size = nothing, flops = nothing))
end

function _get_original_kkt_matrix(madnlp::MadNLP.MadNLPSolver)
    matrix = madnlp.kkt.linear_solver.csc
    rhs = MadNLP.primal_dual(madnlp.p)
    return matrix, rhs
end

function _get_schur_complement(madnlp::MadNLP.MadNLPSolver)
    solver = madnlp.kkt.linear_solver
    matrix = solver.reduced_solver.csc
    rhs = MadNLP.primal_dual(madnlp.p)
    dim = solver.csc.n
    pivot_dim = length(solver.pivot_indices)
    index_set = Set(solver.pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:dim)
    orig_rhs_reduced = rhs[reduced_indices]
    orig_rhs_pivot = rhs[solver.pivot_indices]
    P = solver.pivot_indices
    R = reduced_indices
    A = solver.csc[R, R]
    B = solver.csc[P, R] + solver.csc[R, P]'
    temp = copy(orig_rhs_pivot)
    MadNLP.solve!(solver.pivot_solver, temp)
    rhs_reduced = orig_rhs_reduced - B' * temp
    return matrix, rhs_reduced
end

function _get_pivot_matrix(madnlp::MadNLP.MadNLPSolver)
    solver = madnlp.kkt.linear_solver
    matrix = solver.pivot_solver.csc
    rhs = MadNLP.primal_dual(madnlp.p)
    dim = solver.csc.n
    pivot_dim = length(solver.pivot_indices)
    index_set = Set(solver.pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:dim)
    orig_rhs_reduced = rhs[reduced_indices]
    orig_rhs_pivot = rhs[solver.pivot_indices]
    P = solver.pivot_indices
    R = reduced_indices
    A = solver.csc[R, R]
    B = solver.csc[P, R] + solver.csc[R, P]'
    temp = copy(orig_rhs_pivot)
    MadNLP.solve!(solver.pivot_solver, temp)
    rhs_reduced = orig_rhs_reduced - B' * temp
    MadNLP.solve!(solver.reduced_solver, rhs_reduced)
    rhs_pivot = orig_rhs_pivot - B * rhs_reduced
    return matrix, rhs_pivot
end

"""The diagonal block of the original matrix that we *don't* perform
a Schur complement with respect to"""
function _get_A(madnlp::MadNLP.MadNLPSolver)
    solver = madnlp.kkt.linear_solver
    dim = solver.csc.n
    index_set = Set(solver.pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:dim)
    R = reduced_indices
    A = solver.csc[R, R]
    # There is no RHS that can be meaningfully associated with A
    return A, nothing
end

"""The off-diagonal block of the original matrix"""
function _get_B(madnlp::MadNLP.MadNLPSolver)
    solver = madnlp.kkt.linear_solver
    dim = solver.csc.n
    index_set = Set(solver.pivot_indices)
    reduced_indices = filter(i -> !(i in index_set), 1:dim)
    P = solver.pivot_indices
    R = reduced_indices
    A = solver.csc[R, R]
    B = solver.csc[P, R] + solver.csc[R, P]'
    # There is no RHS that can be meaningfully associated with B
    return B, nothing
end

MATRIX_GETTER = Dict(
    "original" => _get_original_kkt_matrix,
    "schur" => _get_schur_complement,
    "pivot" => _get_pivot_matrix,
    "A" => _get_A,
    "B" => _get_B,
)

function factorize_and_solve_model(
    model::JuMP.Model,
    formulation,
    Solver::Type;
    opt::Union{MadNLP.AbstractOptions,Nothing} = nothing,
    nsamples::Int = 1,
    silent = false,
    return_solver_data = false,
    matrix_type = "original",
    return_runtime_breakdown = false,
)
    _t = time()
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    madnlp_options = Dict{Symbol,Any}(
        # TODO: Can I use MA86 to speed this up? I would need to pass ReducedSolver opts.
        :ReducedSolver => MadNLPHSL.Ma57Solver,
        :PivotSolver => BlockTriangularSolver,
        :pivot_indices => pivot_indices,
        :pivot_solver_opt => pivot_solver_opt,
    )
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Parse formulation")

    if opt === nothing && Solver === SchurComplementSolver
        opt_linear_solver = SchurComplementOptions(; madnlp_options...)
    elseif opt === nothing
        opt_linear_solver = MadNLP.default_options(Solver)
    else
        opt_linear_solver = opt
    end
    _t = time()
    # MadNLP must always use the same linear solver, otherwise the matrices
    # we end up with will be slightly different between different solvers.
    #
    # We initialize MadNLP with max_iter = 0, then proceed with the solve one
    # iteration at a time up to the number of samples specified.
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(
        nlp;
        tol = 1e-6,
        print_level = silent ? MadNLP.ERROR : MadNLP.TRACE,
        max_iter = 0,
        linear_solver = SchurComplementSolver,
        madnlp_options...,
    )
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Initialize MadNLP")

    _t = time()

    # Convert KKT into the matrix we actually want to study here
    matrix, _ = MATRIX_GETTER[matrix_type](madnlp)

    if Solver === SchurComplementSolver && matrix_type != "original"
        error("Can't use SchurComplementSolver with anything but the original KKT matrix")
    end
    linear_solver = Solver(matrix; opt = opt_linear_solver)
    t_init = time() - _t; println("[$(@sprintf("%1.2f", t_init))] Initialize linear solver")
    if return_runtime_breakdown && Solver !== SchurComplementSolver
        error("Can't return a runtime breakdown unless we're using SchurComplementSolver")
    end

    if !silent
        println(MadNLP.introduce(linear_solver))
        display(linear_solver.csc)
    end

    # We will use these to more efficiently compute residuals later.
    # We do this here to amortize the cost across many iterations.
    full_matrix, tril_to_full_view = MadNLP.get_tril_to_full(linear_solver.csc)

    results = []
    for i in 1:nsamples
        println("SAMPLE = $i")
        # Solve with MadNLP up to max_iter
        MadNLP.regular!(madnlp)
        # *After the solve* advance the iteration count
        madnlp.opt.max_iter += 1
        # If we solve the problem before the final sample, all the following
        # matrices will be identical, biasing our "distribution".
        @assert i == nsamples || madnlp.status != MadNLP.SOLVE_SUCCEEDED

        kkt_matrix = MadNLP.get_kkt(madnlp.kkt)
        kkt_rhs = MadNLP.primal_dual(madnlp.p)
        matrix, rhs = MATRIX_GETTER[matrix_type](madnlp)
        # Update matrix in the linear solver. If this is the original KKT matrix, this is a no-op
        linear_solver.csc.nzval .= matrix.nzval

        if return_runtime_breakdown
            tfact_init = (;
                reduced = linear_solver.timer.factorize.reduced,
                pivot = linear_solver.timer.factorize.pivot,
                solve = linear_solver.timer.factorize.solve,
                multiply = linear_solver.timer.factorize.multiply,
            )
            tsolve_init = (;
                schur = linear_solver.timer.solve_timer.solve_schur,
                pivot = linear_solver.timer.solve_timer.solve_pivot,
                rhs = linear_solver.timer.solve_timer.compute_rhs,
            )
        end

        _t = time()
        MadNLP.factorize!(linear_solver)
        inertia = MadNLP.inertia(linear_solver)
        npos, nzero, nneg = inertia
        t_factorize = time() - _t

        # I want to collect the solver return code for factorization, not backsolve.
        if return_solver_data
            solver_data = SOLVER_DATA_GETTER[Solver](linear_solver)
        else
            solver_data = (;)
        end

        sol = copy(rhs)
        _t = time()
        MadNLP.solve!(linear_solver, sol)
        # TODO: Allow specification of these iterative refinement parameters
        refine_res = refine!(
            sol,
            linear_solver,
            rhs,
            max_iter = 20,
            tol = 1e-5;
            full_matrix,
            tril_to_full_view,
        )
        t_solve = time() - _t

        full_matrix = fill_upper_triangle(matrix)
        residual = maximum(abs.(full_matrix * sol - rhs))
        println("residual = $residual")

        res = (;
            dim = matrix.m,
            nnz = SparseArrays.nnz(matrix),
            t_init,
            t_factorize,
            t_solve,
            nneg_eig = nneg,
            residual,
            refine_success = refine_res.success,
            refine_iter = refine_res.iterations,
        )
        res = merge(res, solver_data)
        if return_runtime_breakdown
            # These are the time taken in each category for the most recent solve
            factorize_buckets = [
                linear_solver.timer.factorize.reduced - tfact_init.reduced,
                linear_solver.timer.factorize.pivot - tfact_init.pivot,
                linear_solver.timer.factorize.solve + linear_solver.timer.factorize.multiply - tfact_init.solve - tfact_init.multiply,
            ]
            backsolve_buckets = [
                linear_solver.timer.solve_timer.solve_schur - tsolve_init.schur,
                linear_solver.timer.solve_timer.solve_pivot - tsolve_init.pivot,
                linear_solver.timer.solve_timer.compute_rhs - tsolve_init.rhs,
            ]
            breakdown = (;
                factorize_schur = factorize_buckets[1],
                factorize_pivot = factorize_buckets[2],
                construct_schur = factorize_buckets[3],
                other_factorize = t_factorize - sum(factorize_buckets),
                compute_resid = refine_res.t_resid,
                solve_schur = backsolve_buckets[1],
                solve_pivot = backsolve_buckets[2],
                compute_rhs = backsolve_buckets[3],
                other_backsolve = t_solve - refine_res.t_resid - sum(backsolve_buckets),
            )
            res = merge(res, breakdown)
        end
        if !silent
            println("SAMPLE $i RESULTS:")
            println(res)
        end
        push!(results, res)
    end
    return results
end

function get_matrix_structure(
    model::JuMP.Model,
    formulation;
    matrix_type = "original",
)
    _t = time()
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    # We have to use SchurComplementSolver so we can construct all the submatrices
    madnlp_options = Dict{Symbol,Any}(
        # TODO: Can I use MA86 to speed this up? I would need to pass ReducedSolver opts.
        :ReducedSolver => MadNLPHSL.Ma57Solver,
        :PivotSolver => BlockTriangularSolver,
        :pivot_indices => pivot_indices,
        :pivot_solver_opt => pivot_solver_opt,
    )
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Parse formulation")

    _t = time()
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(
        nlp;
        tol = 1e-6,
        print_level = MadNLP.TRACE,
        max_iter = 0,
        linear_solver = SchurComplementSolver,
        madnlp_options...,
    )
    MadNLP.initialize!(madnlp)
    kkt_system = madnlp.kkt
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Initialize MadNLP")

    matrix, _ = MATRIX_GETTER[matrix_type](madnlp)
    nrow = matrix.m
    ncol = matrix.n
    nnz = SparseArrays.nnz(matrix)
    return (; nrow, ncol, nnz)
end
