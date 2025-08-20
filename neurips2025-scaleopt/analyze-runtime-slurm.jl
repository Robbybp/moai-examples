using Printf
using Distributed, SlurmClusterManager
addprocs(SlurmManager())
@everywhere println("ID: $(myid())")
@everywhere println("Hostname: $(gethostname())")

@everywhere include("analyze-runtime.jl")

# TODO: Does global data need to be defined @everywhere?
#@everywhere begin
#end

devices = Dict(
    :full_space => ["cpu"],
    :vector_nonlinear_oracle => ["cpu", "cuda"],
)

model_names = ["mnist"]
# TODO: These NNs will depend on the model. We will need to look them up.
fnames = [
    "mnist-tanh128nodes4layers.pt",
    "mnist-tanh512nodes4layers.pt",
    "mnist-tanh1024nodes4layers.pt",
    "mnist-tanh2048nodes4layers.pt",
    "mnist-tanh4096nodes4layers.pt",
    "mnist-sigmoid8192nodes4layers.pt",
]
# In this experiment, the only "reduced-space" we care about is VNO.
formulations = [
    :full_space,
    :vector_nonlinear_oracle,
]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")
fpaths = map(f -> joinpath(nn_dir, f), fnames)

NSAMPLES = 1

#models = []
#data = []

inputs = [
    (
        model_name,
        fpath,
        formulation,
        device,
        sample,
    )
    for model_name in model_names
    for fpath in fpaths
    for formulation in formulations
    for device in devices[formulation]
    for sample in 1:NSAMPLES
]
inputs = [(i, input...) for (i, input) in enumerate(inputs)]
n_elements = length(inputs)
println("Running distributed sweep with $n_elements elements")
results_dir = get_results_dir()
@sync @distributed for (i, model_name, fpath, formulation, device, sample) in inputs
    println("SWEEP ELEMENT $i")
    println("----------------")
    println("Model:       $model_name")
    println("NN:          $fpath")
    println("Formulation: $formulation")
    println("Device:      $device")
    println("Sample:      $sample")
    args = (; index = i, model = model_name, NN = basename(fpath), formulation, device, sample)
    _t = time()
    model = MODEL_GETTER[model_name](
        fpath;
        device = device,
        sample_index = sample,
        FORMULATION_TO_KWARGS[formulation]...,
    )
    t_build_total = time() - _t
    println("Model build time: $t_build_total")

    results = solve_model_with_ipopt(model)
    info = merge(args, results, (; t_build_total))

    df = DataFrames.DataFrame([info])
    fname = "runtime-$i.csv"
    fpath = joinpath(results_dir, fname)
    println("Writing results for the following inputs to $fpath")
    println("Model:       $model_name")
    println("NN:          $fpath")
    println("Formulation: $formulation")
    println("Device:      $device")
    println("Sample:      $sample")
    CSV.write(fpath, df)
end

dfs = []
for i in 1:n_elements
    fname = "runtime-$(@sprintf("%02d", i)).csv"
    fpath = joinpath(results_dir, fname)
    if isfile(fpath)
        df = DataFrames.DataFrame(CSV.File(fpath))
        push!(dfs, df)
    else
        @warn "$fpath does not exist"
    end
end
df = reduce(vcat, dfs)
tabledir = get_table_dir()
fname = "runtime.csv"
fpath = joinpath(tabledir, fname)
println("Writing combined results to $fpath")
CSV.write(fpath, df)
