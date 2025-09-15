"""This script solves the KKT system at the initial guess with several linear solvers.
The purpose is to justify our comparison against MA57.
"""

using Printf

import CSV
import DataFrames
import JuMP
import MadNLP
import MadNLPHSL

include("../config.jl")
include("localconfig.jl")

include("solve.jl")

include("../model-getter.jl")
include("nn-getter.jl")

model_names = [
    "mnist",
    "scopf",
]

linear_solvers = [
    #SchurComplementSolver,
    #MadNLPHSL.Ma27Solver,
    MadNLPHSL.Ma57Solver,
    MadNLPHSL.Ma86Solver,
    #MadNLPHSL.Ma97Solver,
]

OPT_LOOKUP = Dict(
    # Metis or exact minimum degree croak on these matrices.
    MadNLPHSL.Ma57Solver => MadNLPHSL.Ma57Options(; ma57_pivot_order = 2), # In MA57, 2=AMD
    MadNLPHSL.Ma86Solver => MadNLPHSL.Ma86Options(; ma86_order = MadNLPHSL.AMD, ma86_num_threads = 1),
    MadNLPHSL.Ma97Solver => MadNLPHSL.Ma97Options(; ma97_order = MadNLPHSL.AMD, ma97_num_threads = 1),
)
function _get_opt(SolverType)
    return get(OPT_LOOKUP, SolverType, nothing)
end

# I don't intend to sample multiple primal values in this script
#nsamples = 10

for model_name in model_names
    # Precompile on a small instance
    println("PRECOMPILING WITH MODEL: $model_name")
    precompile_nnfname = MODEL_TO_PRECOMPILE_NN[model_name]
    precompile_nnfpath = joinpath(get_nn_dir(), precompile_nnfname)
    precompile_model, precompile_formulation = MODEL_GETTER[model_name](precompile_nnfpath)
    for SolverType in linear_solvers
        factorize_and_solve_model(precompile_model, precompile_formulation, SolverType; silent = true)
    end
    println("DONE PRECOMPILING WITH MODEL: $model_name")
end

n_nns = 3

data = []
for model_name in model_names
    # We only test on the first two NNs, as the larger ones hit errors in the
    # symbolic factorization phase.
    # TODO: This may be fixable using Int64, but this is not necessarily a priority
    # for now.
    NNs = MODEL_TO_NNS[model_name][1:n_nns]
    for nnfname in NNs
        nnfpath = joinpath(get_nn_dir(), nnfname)
        # Here, sample index refers to the sample of the model itself, not the primal
        # variable values within the model.
        model, formulation = MODEL_GETTER[model_name](nnfpath; sample_index = 1)
        for SolverType in linear_solvers
            println("Starting trial with following parameters:")
            println("Model:      $model_name")
            println("NN:         $nnfname")
            println("Solver:     $SolverType")
            inputs = (;
                model = model_name,
                nn = nnfname,
                solver = String(Symbol(SolverType)),
            )
            results = factorize_and_solve_model(
                model,
                formulation,
                SolverType;
                opt = _get_opt(SolverType),
            )
            println("Finished trial with following parameters:")
            println("Model:      $model_name")
            println("NN:         $nnfname")
            println("Solver:     $SolverType")
            for (i, res) in enumerate(results)
                println("Sample $i results:")
                println(res)
                info = merge(inputs, (; sample = i), res)
                push!(data, info)
            end
        end
    end
end

df = DataFrames.DataFrame(data)
println(df)
tabledir = get_table_dir()
fname = "linear-solvers.csv"
fpath = joinpath(tabledir, fname)
println("Writing results to $fpath")
CSV.write(fpath, df)
