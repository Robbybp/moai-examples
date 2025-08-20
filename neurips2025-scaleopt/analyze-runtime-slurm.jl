using Printf
using Distributed, SlurmClusterManager
addprocs(SlurmManager())
@everywhere println("ID: $(myid())")
@everywhere println("Hostname: $(gethostname())")

@everywhere include("analyze-runtime.jl")

# Does this global data need to be declared @everywhere?
include("setup-compare-formulations.jl")
results_dir = get_results_dir()

# Only run the sweep if we're running this file directly
if abspath(PROGRAM_FILE) == @__FILE__

println("Running distributed sweep with $n_elements elements")
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

include("collect-slurm-runtimes.jl")

end
