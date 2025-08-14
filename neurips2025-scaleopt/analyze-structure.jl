"""This script currently runs a sweep and displays a DataFrame of structural
information about many models.

TODO:
- Put model getting utilities in a separate file
- Put the sweep and printing in a separate file, leaving this file with only the
  core functionality

"""

import JuMP
import Ipopt
import DataFrames

# TODO: Move model getter to config file or something
include("../adversarial-image.jl")
function _get_adversarial_image_model(nnfile::String; kwargs...)
    image_index = 7
    adversarial_label = 1
    threshold = 0.6
    model, _ = get_adversarial_model(nnfile, image_index, adversarial_label, threshold; kwargs...)
    return model
end
MODEL_GETTER = Dict(
    "mnist" => _get_adversarial_image_model,
)

FORMULATION_TO_KWARGS = Dict(
    :full_space => Dict(),
    :reduced_space => Dict(:reduced_space => true),
    :gray_box => Dict(:gray_box => true),
    :vector_nonlinear_oracle => Dict(:vector_nonlinear_oracle => true),
)

function get_model_structure(model::JuMP.Model)
    nvar = length(JuMP.all_variables(model))
    ncon = length(JuMP.all_constraints(model; include_variable_in_set_constraints = false))
    JuMP.set_optimizer(model, Ipopt.Optimizer)
    # This is necessary to get nonzeros due to linear, quadratic, and nonlinear parts of the model
    ipopt = JuMP.unsafe_backend(model)
    MOIExt = Base.get_extension(Ipopt, :IpoptMathOptInterfaceExt)
    @assert MOIExt !== nothing
    MOIExt._setup_model(ipopt)
    hessian_structure = MOI.hessian_lagrangian_structure(ipopt)
    jacobian_structure = MOI.jacobian_structure(ipopt)
    jnnz = length(Set(jacobian_structure))
    hnnz = length(Set(hessian_structure))
    return (; nvar, ncon, jnnz, hnnz)
end

models = ["mnist"]
# TODO: These NNs will depend on the model. We will need to look them up.
fnames = [
    "mnist-tanh128nodes4layers.pt",
    #"mnist-tanh512nodes4layers.pt",
    #"mnist-tanh1024nodes4layers.pt",
    #"mnist-tanh2048nodes4layers.pt",
    #"mnist-tanh4096nodes4layers.pt",
    #"mnist-tanh8192nodes4layers.pt",
]
formulations = [
    :full_space,
    :reduced_space,
    :gray_box,
    :vector_nonlinear_oracle,
]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")
fpaths = map(f -> joinpath(nn_dir, f), fnames)

data = []
for model_name in models
    for fpath in fpaths
        for formulation in formulations
            model = MODEL_GETTER[model_name](fpath; FORMULATION_TO_KWARGS[formulation]...)
            info = get_model_structure(model)
            push!(data, info)
        end
    end
end
df = DataFrames.DataFrame(data)
println(df)
