using Printf

import JuMP
import MadNLP
import MadNLPHSL

include("../linalg.jl")
include("../btsolver.jl")
include("../formulation.jl")
include("../nlpmodels.jl")
include("../kkt-partition.jl")

function factorize_and_solve_model(
    model::JuMP.Model,
    formulation,
    Solver::Type;
    opt::Union{MadNLP.AbstractOptions,Nothing} = nothing,
    nsamples::Int = 1,
    silent = false,
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

    # For some reason, using kkt_system.linear_solver yields an error when I factorize...
    _t = time()
    linear_solver = Solver(kkt_matrix; opt = opt_linear_solver)
    t_init = time() - _t; println("[$(@sprintf("%1.2f", t_init))] Initialize linear solver")

    if !silent
        println(MadNLP.introduce(linear_solver))
        display(linear_solver.csc)
    end

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

        # This should be a no-op, but just to be sure that this has been updated:
        linear_solver.csc.nzval .= MadNLP.get_kkt(madnlp.kkt).nzval

        _t = time()
        MadNLP.factorize!(linear_solver)
        inertia = MadNLP.inertia(linear_solver)
        npos, nzero, nneg = inertia
        t_factorize = time() - _t

        # Get the KKT system RHS from MadNLP
        rhs = MadNLP.primal_dual(madnlp.p)
        sol = copy(rhs)
        _t = time()
        MadNLP.solve!(linear_solver, sol)
        # TODO: Allow specification of these iterative refinement parameters
        refine_res = refine!(sol, linear_solver, rhs, max_iter = 20, tol = 1e-5)
        t_solve = time() - _t

        full_kkt = fill_upper_triangle(kkt_matrix)
        residual = maximum(abs.(full_kkt * sol - rhs))
        println("residual = $residual")

        res = (;
            t_init,
            t_factorize,
            t_solve,
            nneg_eig = nneg,
            residual,
            refine_success = refine_res.success,
            refine_iter = refine_res.iterations,
        )
        if !silent
            println("SAMPLE $i RESULTS:")
            println(res)
        end
        push!(results, res)
    end
    return results
end
