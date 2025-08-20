# Just use whatever Python we find on the path, whether or not it is from conda
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

import MathOptAI
import DataFrames

include("../pytorch.jl")
include("model-getter.jl")

_isinstance(a, b) = Bool(PythonCall.pybuiltins.isinstance(a, b))

function _get_input_dimension(nn)::Int
    torch = PythonCall.pyimport("torch")
    if _isinstance(nn, torch.nn.Linear)
        return PythonCall.pyconvert(Int, nn.in_features)
    elseif _isinstance(nn, torch.nn.Sequential)
        # HACK: I'm just getting the input dimen of the first layer...
        children = PythonCall.pybuiltins.list(nn.children())
        return _get_input_dimension(children[0])
    end
    layers = PythonCall.pybuiltins.list(nn.children())
    i = 0 # Using base-0 Python indexing here.
    while i < length(layers) && !_isinstance(layers[i], torch.nn.Linear)
        # Advance counter until we find a linear layer
        i += 1
    end
    if i == length(layers)
        error("NN does not have a well-defined input dimension")
    end
    return PythonCall.pyconvert(Int, layers[i].in_features)
end

function _get_neurons_per_layer(layer; first = true)
    torch = PythonCall.pyimport("torch")
    neurons_per_layer = []
    if _isinstance(layer, torch.nn.Sequential)
        for (i, l) in enumerate(layer.children())
            # Only send first=true to the first child.
            # This is not strictly correct. What if the first linear layer appears second...
            # The true solution is to flatten all layers first...
            first &= (i == 1)
            append!(neurons_per_layer, _get_neurons_per_layer(l; first = first))
        end
    elseif _isinstance(layer, torch.nn.Linear)
        if first
            push!(neurons_per_layer, PythonCall.pyconvert(Int, layer.in_features))
        end
        push!(neurons_per_layer, PythonCall.pyconvert(Int, layer.out_features))
    end
    return neurons_per_layer
end

function _get_activations(layer)
    torch = PythonCall.pyimport("torch")
    activations = Symbol[]
    if _isinstance(layer, torch.nn.Sequential)
        for l in layer.children()
            append!(activations, _get_activations(l))
        end
    elseif _isinstance(layer, torch.nn.Tanh)
        push!(activations, :Tanh)
    elseif _isinstance(layer, torch.nn.ReLU)
        push!(activations, :ReLU)
    elseif _isinstance(layer, torch.nn.Sigmoid)
        push!(activations, :Sigmoid)
    elseif _isinstance(layer, torch.nn.Softmax)
        push!(activations, :SoftMax)
    elseif _isinstance(layer, torch.nn.Softplus)
        push!(activations, :SoftPlus)
    end
    return unique(activations)
end

function _get_counts(array)
    counts = Dict()
    for item in array
        if item in keys(counts)
            counts[item] += 1
        else
            counts[item] = 1
        end
    end
    return counts
end

function analyze_nn(fpath::String)
    torch = PythonCall.pyimport("torch")
    nn = torch.load(fpath, weights_only = false, map_location="cpu")
    n_inputs = _get_input_dimension(nn)
    x = torch.tensor(ones(n_inputs))
    y = nn(x) # By the MathOptAI convention, this must be a flat vector of outputs (IIRC)
    n_outputs = length(y)
    n_parameters = count_nn_parameters(fpath)
    activations = _get_activations(nn)
    neurons_per_layer = _get_neurons_per_layer(nn)
    counts = _get_counts(neurons_per_layer)
    # Not all layers have the same size, so I use the mode of the size-per-layer
    # as the number that I report.
    sorted_nperlayer = sort(unique(neurons_per_layer), by = i -> counts[i], rev = true)
    n_neurons_per_layer = sorted_nperlayer[1]
    n_layers = length(neurons_per_layer)
    n_neurons = sum(neurons_per_layer)
    info = (; n_inputs, n_outputs, n_neurons, n_layers, n_neurons_per_layer, n_parameters, activations)
    return info
end

model_names = ["mnist", "scopf"]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")
nn_data = []
for model_name in model_names
    for fname in MODEL_TO_NNS[model_name]
        fpath = joinpath(nn_dir, fname)
        res = analyze_nn(fpath)
        inputs = (; fname)
        res = merge(inputs, res)
        push!(nn_data, res)
    end
end
df = DataFrames.DataFrame(nn_data)
println(df)
