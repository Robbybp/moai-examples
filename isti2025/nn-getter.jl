MODEL_TO_NNS = Dict(
    "mnist" => [
        "mnist-relu128nodes4layers.pt",
        "mnist-relu512nodes4layers.pt",
        "mnist-relu1024nodes4layers.pt",
        "mnist-relu2048nodes4layers.pt",
        #"mnist-relu4096nodes4layers.pt",
        #"mnist-sigmoid8192nodes4layers.pt",
    ],
    "scopf" => [
        joinpath("scopf", "100nodes3layers.pt"),
        joinpath("scopf", "500nodes5layers.pt"),
        joinpath("scopf", "1000nodes7layers.pt"),
        #joinpath("scopf", "2000nodes20layers.pt"),
        #joinpath("scopf", "4000nodes40layers.pt"),
    ],
)

MODEL_TO_PRECOMPILE_NN = Dict(
    "mnist" => "mnist-tanh128nodes4layers.pt",
    "scopf" => joinpath("scopf", "100nodes3layers.pt"),
)
