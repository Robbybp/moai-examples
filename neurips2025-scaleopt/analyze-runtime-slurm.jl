using Printf
using Distributed, SlurmClusterManager
addprocs(SlurmManager())
@everywhere println("ID: $(myid())")
@everywhere println("Hostname: $(gethostname())")

@everywhere ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
# Does this global data need to be declared @everywhere?
@everywhere include("setup-compare-formulations.jl")

@everywhere include("analyze-runtime.jl")
@everywhere include("../pytorch.jl")
results_dir = get_results_dir()

# Only run the sweep if we're running this file directly
if abspath(PROGRAM_FILE) == @__FILE__

println("Running distributed sweep with $n_elements elements")
@sync @distributed for (i, model_name, fname, formulation, device, sample) in inputs
    local fpath = joinpath(nn_dir, fname)
    println("SWEEP ELEMENT $i")
    println("----------------")
    println("Model:       $model_name")
    println("NN:          $fpath")
    println("Formulation: $formulation")
    println("Device:      $device")
    println("Sample:      $sample")
    println("CPU:         $(Sys.cpu_info()[1].model)")
    println("GPU:         $(get_pytorch_device_name())")

    args = (;
        index = i,
        model = model_name,
        NN = basename(fpath),
        formulation,
        device,
        cpu = Sys.cpu_info()[1].model,
        gpu = get_pytorch_device_name(),
        sample,
    )
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
    fname = "runtime-$(@sprintf("%02d", i)).csv"
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
