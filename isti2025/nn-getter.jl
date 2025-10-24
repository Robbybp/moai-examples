MODEL_TO_NNS = Dict(
    "mnist" => [
        #"mnist-tanh128nodes4layers.pt",
        "mnist-tanh512nodes4layers.pt",
        "mnist-tanh1024nodes4layers.pt",
        "mnist-tanh2048nodes4layers.pt",
        #"mnist-relu128nodes4layers.pt",
        #"mnist-relu512nodes4layers.pt",
        #"mnist-relu1024nodes4layers.pt",
        #"mnist-relu2048nodes4layers.pt",
        #"mnist-relu4096nodes4layers.pt",
        #"mnist-sigmoid8192nodes4layers.pt",
    ],
    "scopf" => [
        #joinpath("scopf", "100nodes3layers.pt"),
        joinpath("scopf", "500nodes5layers.pt"),
        joinpath("scopf", "1000nodes7layers.pt"),
        joinpath("scopf", "1500nodes10layers.pt"),
        #joinpath("scopf", "2000nodes20layers.pt"),
        #joinpath("scopf", "4000nodes40layers.pt"),
        #joinpath("scopf", "relu100nodes3layers.pt"),
        #joinpath("scopf", "relu500nodes5layers.pt"),
        #joinpath("scopf", "relu1000nodes7layers.pt"),
    ],
    "lsv" => [
        joinpath("lsv", "118_bus", "118_bus_128node.pt"),
        joinpath("lsv", "118_bus", "118_bus_512node.pt"),
        joinpath("lsv", "118_bus", "118_bus_2048node.pt"),
    ],
)

MODEL_TO_PRECOMPILE_NN = Dict(
    "mnist" => "mnist-tanh128nodes4layers.pt",
    "scopf" => joinpath("scopf", "100nodes3layers.pt"),
    "lsv" => joinpath("lsv", "118_bus", "118_bus_32node.pt"),
)
