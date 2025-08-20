# TODO: Move model getter to config file or something
include("../adversarial-image.jl")
include("../scopf/model.jl")

"""Model functions are expected to accept the following parameters:
- NN (path to .pt file)
- The following parameters of add_predictor
  - device = "cpu"
  - reduced_space = false
  - gray_box = false
  - vector_nonlinear_oracle = false
- sample_index = 1
"""

# Index in the test data
SAMPLE_TO_MNIST_INDEX = [7, 1887, 595, 8624, 445, 5645, 3780, 792, 9680, 8929,
    3611, 7602, 9547, 1128, 5322, 6361, 4045, 2326, 29, 2966, 7191, 7063, 8670,
    8995, 6126, 6803, 8852, 6358, 5198, 4455, 3081, 2742, 221, 1794, 6147,
    3606, 6614, 3165, 985, 5570, 4816, 2314, 9909, 8679, 8817, 9182, 21, 8734,
    6141, 9722, 942, 7462, 8449, 3628, 2667, 6755, 674, 2610, 5975, 5694, 4470,
    1835, 4269, 8454, 9892, 7926, 5537, 6521, 3668, 8015, 1215, 9822, 4782,
    895, 6668, 6283, 3296, 8721, 6264, 1708, 1961, 7551, 1402, 6795, 1976, 757,
    6581, 1335, 9380, 6993, 2697, 4825, 8367, 6764, 3419, 3660, 6415, 1146,
    4293, 4266]
SAMPLE_TO_MNIST_TARGET = [4, 6, 6, 9, 2, 8, 1, 5, 6, 9, 4, 0, 7, 3, 0, 7, 3, 7,
    0, 1, 0, 0, 8, 9, 1, 4, 0, 2, 5, 9, 3, 2, 7, 9, 5, 1, 6, 6, 1, 8, 7, 9, 6,
    7, 8, 2, 9, 7, 0, 7, 7, 6, 0, 8, 7, 2, 9, 6, 0, 6, 3, 1, 4, 5, 9, 6, 2, 9,
    7, 3, 1, 3, 7, 3, 6, 5, 5, 1, 7, 1, 8, 4, 7, 3, 7, 7, 0, 5, 2, 7, 9, 0, 5,
    3, 7, 3, 5, 4, 1, 4]
SAMPLE_TO_ADVERSARIAL_LABEL = [ADVERSARIAL_LABEL_LOOKUP[t] for t in SAMPLE_TO_MNIST_TARGET]

function _get_adversarial_image_model(nnfile::String; sample_index = 1, kwargs...)
    # Note that this is the index in the test data
    @assert 1 <= sample_index <= 100
    image_index = SAMPLE_TO_MNIST_INDEX[sample_index]
    adversarial_label = SAMPLE_TO_ADVERSARIAL_LABEL[sample_index]
    threshold = 0.6
    # These were the original defaults. The original image had target 4. Note that we
    # now map 4 to an adversarial target of 9.
    #image_index = 7
    #adversarial_label = 1
    @assert 1 <= image_index <= 10_000
    @assert 0 <= adversarial_label <= 9
    @assert 0.0 <= threshold <= 1.0
    model, _ = get_adversarial_model(nnfile, image_index, adversarial_label, threshold; kwargs...)
    return model
end

import PowerModels
import PowerModelsSecurityConstrained as PMSC
import MathOptAI as MOAI
function _get_scopf_model(nnfile::String; sample_index = 1, kwargs...)
    # TODO: adjust loads randomly with a seed depending on the sample
    @assert sample_index == 1
    files = (;
        raw = joinpath(SCOPF_DIR, "hawaii37.raw"),
        con = joinpath(SCOPF_DIR, "hawaii37-1gen.con"),
        rop = joinpath(SCOPF_DIR, "hawaii37.rop"),
        inl = joinpath(SCOPF_DIR, "hawaii37.inl"),
    )
    input_data = PMSC.parse_c1_files(files.con, files.inl, files.raw, files.rop)
    pmdata = PMSC.build_c1_pm_model(input_data)
    stability_surrogate = MOAI.PytorchModel(nnfile)
    surrogate_params = Dict{Symbol,Any}(kwargs)
    pm, info = build_scopf(pmdata; stability_surrogate, surrogate_params)
    return pm.model
end
MODEL_GETTER = Dict(
    "mnist" => _get_adversarial_image_model,
    "scopf" => _get_scopf_model,
)

FORMULATION_TO_KWARGS = Dict(
    :full_space => Dict(),
    :reduced_space => Dict(:reduced_space => true),
    :gray_box => Dict(:gray_box => true, :hessian => true, :reduced_space => false),
    :vector_nonlinear_oracle => Dict(:vector_nonlinear_oracle => true, :hessian => true),
)

MODEL_TO_NNS = Dict(
    "mnist" => [
        "mnist-tanh128nodes4layers.pt",
        "mnist-tanh512nodes4layers.pt",
        "mnist-tanh1024nodes4layers.pt",
        "mnist-tanh2048nodes4layers.pt",
        "mnist-tanh4096nodes4layers.pt",
        #"mnist-sigmoid8192nodes4layers.pt",
    ],
    "scopf" => [
        joinpath("scopf", "100nodes3layers.pt"),
        joinpath("scopf", "500nodes5layers.pt"),
        joinpath("scopf", "1000nodes7layers.pt"),
        joinpath("scopf", "2000nodes20layers.pt"),
        joinpath("scopf", "4000nodes40layers.pt"),
    ],
)
