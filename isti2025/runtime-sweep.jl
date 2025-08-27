using Printf

import CSV
import DataFrames
import JuMP
import MadNLP
import MadNLPHSL

include("../config.jl")
include("localconfig.jl")
include("../linalg.jl")
include("../btsolver.jl")
include("../formulation.jl")
include("../nlpmodels.jl")
include("../kkt-partition.jl")
include("../model-getter.jl")
include("nn-getter.jl")

# This is only necessary for update_kkt!, which really should be moved to
# another file.
include("../models.jl")

# What should the input be here? At some level, the input needs to
# be a JuMP model. I just need to make sure I'm not duplicating
# work among the different samples.
function factorize_and_solve_model(
    model::JuMP.Model,
    formulation,
    Solver::Type;
    opt::Union{MadNLP.AbstractOptions,Nothing} = nothing,
    nsamples::Int = 1,
)
    @assert nsamples == 1
    if opt === nothing && Solver === SchurComplementSolver
        _t = time()
        pivot_vars, pivot_cons = get_vars_cons(formulation)
        pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
        pivot_indices = convert(Vector{Int32}, pivot_indices)
        blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
        pivot_solver_opt = BlockTriangularOptions(; blocks)
        opt = SchurComplementOptions(;
            ReducedSolver = MadNLPHSL.Ma57Solver,
            PivotSolver = BlockTriangularSolver,
            pivot_indices,
            pivot_solver_opt,
        )
        dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Parse formulation")
    elseif opt === nothing
        opt = MadNLP.default_options(Solver)
    end
    _t = time()
    nlp, kkt_system, kkt_matrix = get_kkt(
        model;
        Solver,
        opt_linear_solver = opt,
    )
    # For some reason, using kkt_system.linear_solver yields an error when I factorize...
    linear_solver = Solver(kkt_matrix; opt)
    # I could get the initialize time from SchurComplementTimer, but this doesn't work
    # for MA57
    t_init = time() - _t; println("[$(@sprintf("%1.2f", t_init))] Initialize KKT System (and linear solver)")

    println(MadNLP.introduce(linear_solver))
    display(linear_solver.csc)

    # TODO: Initialize with somewhat random primal/dual values
    # TODO: Loop over number of samples
    t_factorize = 0.0
    t_solve = 0.0
    max_abs_residuals = []
    for i in 1:nsamples
        local _t = time()
        MadNLP.factorize!(linear_solver)
        t_factorize += time() - _t

        # TODO: use a reasonable RHS by actually evaluating the constraints and
        # Lagrangian?
        #rhs = i .* Vector{Float64}(1:kkt_matrix.m)
        rhs = randn(kkt_matrix.m)
        sol = copy(rhs)
        local _t = time()
        MadNLP.solve!(linear_solver, sol)
        t_solve += time() - _t

        # TODO: Compute residual
        full_kkt = fill_upper_triangle(kkt_matrix)
        abs_residual = maximum(abs.(full_kkt * sol - rhs))
        println("residual = $abs_residual")
        push!(max_abs_residuals, abs_residual)
    end
    ave_residual = sum(max_abs_residuals) / length(max_abs_residuals)

    results = (;
        #linear_solver, # For debugging. Or I can just return linear_solver.timer, if that's all I need
        t_init,
        t_factorize,
        t_solve,
        ave_residual,
    )
end

model_names = [
    "mnist",
    "scopf",
]

linear_solvers = [
    SchurComplementSolver,
    MadNLPHSL.Ma57Solver,
]

data = []
for model_name in model_names
    for nnfname in MODEL_TO_NNS[model_name]
        nnfpath = joinpath(get_nn_dir(), nnfname)
        model, formulation = MODEL_GETTER[model_name](nnfpath)
        for SolverType in linear_solvers
            println("Model:  $model_name")
            println("NN:     $nnfname")
            println("Solver: $SolverType")
            inputs = (;
                model = model_name,
                nn = nnfname,
                solver = String(Symbol(SolverType)),
            )
            results = factorize_and_solve_model(model, formulation, SolverType)
            println(results)
            info = merge(inputs, results)
            push!(data, info)
        end
    end
end

df = DataFrames.DataFrame(data)
println(df)
tabledir = get_table_dir()
fname = "runtime.csv"
fpath = joinpath(tabledir, fname)
CSV.write(fpath, df)
