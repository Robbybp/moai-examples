import ArgParse
import Ipopt
import JuMP
import PythonCall
import MathOptAI as MOAI
import MLDatasets

include("adversarial-image.jl")

FILEDIR = dirname(@__FILE__)

settings = ArgParse.ArgParseSettings()
ArgParse.add_arg_table(
    settings,
    "fpath",
    Dict(:help=>"NN .pt file"),
    "--index",
    Dict(
        :help=>"index of image (BASE-1!!!)",
        :default=>7, # TODO: Choose a reasonable default
        :arg_type=>Int,
    ),
    "--label",
    Dict(
        :help=>"Desired adversarial label",
        :default=>7, # TODO: Choose a reasonable default
        :arg_type=>Int,
    ),
    "--threshold",
    Dict(
        :help=>"Threshold for NN output of target label",
        :default=>0.6, # TODO: Choose a reasonable default
        :arg_type=>Float64,
    ),
    "--data-dir",
    Dict(
        :help=>"Directory to save data",
        :default=>joinpath(FILEDIR, "data"),
    ),
)
args = ArgParse.parse_args(settings)

find_adversarial_image(
    args["fpath"],
    args["index"],
    args["label"],
    args["threshold"],
)
