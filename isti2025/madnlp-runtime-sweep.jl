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
    "lsv",
]

linear_solvers = Dict(
    "mnist" => [
        SchurComplementSolver,
        MadNLPHSL.Ma57Solver,
    ],
    "scopf" => [
        SchurComplementSolver,
        MadNLPHSL.Ma86Solver,
    ],
    "lsv" => [
        SchurComplementSolver,
        MadNLPHSL.Ma86Solver,
    ],
)

OPT_LOOKUP = Dict(
    # Metis or exact minimum degree croak on these matrices.
    MadNLPHSL.Ma57Solver => MadNLPHSL.Ma57Options(; ma57_pivot_order = 2), # In MA57, 2=AMD
    MadNLPHSL.Ma86Solver => MadNLPHSL.Ma86Options(; ma86_order = MadNLPHSL.AMD, ma86_num_threads = 1),
    MadNLPHSL.Ma97Solver => MadNLPHSL.Ma97Options(; ma97_order = MadNLPHSL.AMD, ma97_num_threads = 1),
)
function _get_opt(SolverType)
    return get(OPT_LOOKUP, SolverType, nothing)
end

nsamples = 10

for model_name in model_names
    # Precompile on a small instance
    println("PRECOMPILILING WITH MODEL: $model_name")
    precompile_nnfname = MODEL_TO_PRECOMPILE_NN[model_name]
    precompile_nnfpath = joinpath(get_nn_dir(), precompile_nnfname)
    precompile_model, precompile_formulation = MODEL_GETTER[model_name](precompile_nnfpath)
    for SolverType in linear_solvers[model_name]
        factorize_and_solve_model(
            precompile_model,
            precompile_formulation,
            SolverType;
            opt = _get_opt(SolverType),
            silent = true,
        )
    end
    println("DONE PRECOMPILILING WITH MODEL: $model_name")
end

data = []
for model_name in model_names
    for nnfname in MODEL_TO_NNS[model_name]
        nnfpath = joinpath(get_nn_dir(), nnfname)
        model, formulation = MODEL_GETTER[model_name](nnfpath; sample_index = 1)
        for SolverType in linear_solvers[model_name]
            println("Starting trial with following parameters:")
            println("Model:      $model_name")
            println("NN:         $nnfname")
            println("Solver:     $SolverType")
            println("Options:    $(_get_opt(SolverType))")
            println("N. samples: $nsamples")
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
                nsamples,
            )
            println("Finished trial with following parameters:")
            println("Model:      $model_name")
            println("NN:         $nnfname")
            println("Solver:     $SolverType")
            println("N. samples: $nsamples")
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
# Note that this overwrites the earlier table of runtime results.
# I don't want to call this madnlp-runtime.csv as that implies that
# we are actually solving with MadNLP.
fname = "runtime.csv"
fpath = joinpath(tabledir, fname)
println("Writing results to $fpath")
CSV.write(fpath, df)
