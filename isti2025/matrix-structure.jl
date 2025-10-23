import Printf
import CSV
import DataFrames: DataFrame

include("../config.jl")
include("localconfig.jl")

# includes get_matrix_structure
include("solve.jl")

include("../model-getter.jl")
include("nn-getter.jl")

model_names = [
    "mnist",
    "scopf",
]

matrices = [
    "original",
    "pivot",
    "schur",
    "A",
    "B",
]

data = []
for model_name in model_names
    # NOTE: We only consider the last (largest) NN in this sweep
    nns = MODEL_TO_NNS[model_name]
    for nnfname in nns
        nnfpath = joinpath(get_nn_dir(), nnfname)
        # Here, sample index refers to the sample of the model itself, not the primal
        # variable values within the model.
        model, formulation = MODEL_GETTER[model_name](nnfpath; sample_index = 1)
        for matrix in matrices
            println("Starting trial with following parameters:")
            println("Model:      $model_name")
            println("NN:         $nnfname")
            println("Matrix:     $matrix")
            inputs = (;
                model = model_name,
                nn = nnfname,
                matrix_type = matrix,
            )
            results = get_matrix_structure(
                model,
                formulation;
                matrix_type = matrix,
            )
            println("Finished trial with following parameters:")
            println("Model:      $model_name")
            println("NN:         $nnfname")
            println("Matrix:     $matrix")
            println("Results: $results")
            info = merge(inputs, results)
            push!(data, info)
        end
    end
end

df = DataFrame(data)
println(df)
tabledir = get_table_dir()
fname = "matrix-structure.csv"
fpath = joinpath(tabledir, fname)
println("Writing results to $fpath")
CSV.write(fpath, df)
