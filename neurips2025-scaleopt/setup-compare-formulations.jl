"""Global data required for the main parameter sweep, which primarily compares
the full-space and reduced-space formulations for different sizes of NN.
This is used by:
- Serial sweep
- Slurm sweep
- Post-sweep collection

This should be *the only* place where I modify code to alter what parameters
I'm using for this sweep.
"""
# TODO: Should this be moved into localconfig.jl?

# This is where MODEL_TO_NNS is defined
include("../model-getter.jl")

devices = Dict(
    :full_space => ["cpu"],
    :vector_nonlinear_oracle => [
        "cpu",
        "cuda",
    ],
)

model_names = ["mnist", "scopf"]
# In this experiment, the only "reduced-space" we care about is VNO.
formulations = [
    :full_space,
    :vector_nonlinear_oracle,
]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")

NSAMPLES = 1

inputs = [
    (
        model_name,
        fname,
        formulation,
        device,
        sample,
    )
    for model_name in model_names
    for fname in MODEL_TO_NNS[model_name]
    for formulation in formulations
    for device in devices[formulation]
    for sample in 1:NSAMPLES
]
inputs = [(i, input...) for (i, input) in enumerate(inputs)]
n_elements = length(inputs)
