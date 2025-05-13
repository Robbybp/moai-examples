import ArgParse
import Ipopt
import JuMP
import PythonCall
import MathOptAI as MOAI
import MLDatasets

include("config.jl")
include("adversarial-image.jl")
VALID_OPTIMIZERS = keys(OPTIMIZER_LOOKUP)

settings = get_cli_settings()
args = ArgParse.parse_args(settings)

find_adversarial_image(
    args["fpath"],
    args["index"],
    args["label"],
    args["threshold"],
    args["solver"];
    reduced_space = args["reduced-space"],
)
