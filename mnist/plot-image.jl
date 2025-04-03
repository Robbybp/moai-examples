import ArgParse
import Plots
import MLDatasets

function plot_image(x::Matrix; kwargs...)
    return Plots.heatmap(
        x'[size(x, 1):-1:1, :];
        xlims = (1, size(x, 2)),
        ylims = (1, size(x, 1)),
        aspect_ratio = true,
        legend = false,
        xaxis = false,
        yaxis = false,
        kwargs...,
    )
end

#if abspath(PROGRAM_FILE) == @__FILE__
settings = ArgParse.ArgParseSettings()
ArgParse.add_arg_table(
    settings,
    "--index",
    Dict(
        :help=>"index of image (BASE-1!!!)",
        :default=>1, # TODO: Choose a reasonable default
        :arg_type=>Int,
    ),
)
args = ArgParse.parse_args(settings)
index = args["index"]
#end

datadir = joinpath("data", "MNIST", "raw")
test_data = MLDatasets.MNIST(; split = :test, dir = datadir)
image = test_data[index].features
plot = plot_image(image)
# This doesn't really help, because the image disappears right away
#display(plot)
fname = "mnist-$index-test.png"
image_dir = "images"
fpath = joinpath(image_dir, fname)
Plots.savefig(plot, fpath)
