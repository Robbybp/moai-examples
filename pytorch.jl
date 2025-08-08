import PythonCall

function count_nn_parameters(fpath)
    torch = PythonCall.pyimport("torch")
    nn = torch.load(fpath, weights_only = false)
    params = PythonCall.pyconvert(Vector{PythonCall.Core.Py}, nn.parameters())
    params = filter(p -> PythonCall.pyconvert(Bool, p.requires_grad), params)
    return sum(PythonCall.pyconvert(Int, p.numel()) for p in params)
end

function get_pytorch_device_name()
    torch = PythonCall.pyimport("torch")
    if PythonCall.pyconvert(Bool, torch.cuda.is_available())
        dev = torch.cuda.current_device()
        return torch.cuda.get_device_name(dev)
    else
        return nothing
    end
end
