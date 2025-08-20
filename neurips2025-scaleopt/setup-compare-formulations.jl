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
