import Ipopt
import JuMP
import Plots
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import PythonCall
import MathOptAI as MOAI
import MLDatasets

nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
image_index = 7
adversarial_label = 1
# Network is trained so that outputs represent 0-9
adversarial_target_index = adversarial_label + 1
threshold = 0.6

# TODO: make_model function
# - accepts filename, image index, adversary label, and formulation
# - returns JuMP model

# find_adversarial_image function
# - accepts the above, plus solver
# - solves the JuMP model, extracts the pixel values
# - returns a matrix of the pixel values, along with the network's predictions?

predictor = MOAI.PytorchModel(nnfile)

# TODO: Configurable data dir
datadir = joinpath("data", "MNIST", "raw")
test_data = MLDatasets.MNIST(; split = :test, dir = datadir)

length_dim = 28
height_dim = 28

xref = test_data[image_index].features

# TODO: Evaluate the sample image with the network to make sure it is correct
#
torch = PythonCall.pyimport("torch")
nn = torch.load(nnfile, weights_only = false)
input_tensor = torch.tensor(vec(xref))
prediction = nn(input_tensor)
# How can I make sure my Julia and Python matrices are not transposes of each other?
# This is really something that should be documented by the packages that convert the
# images to arrays.
#
# In MLDatasets, images are indexed by width, then height. I.e. they are column-major.
# Fortunately, `vec` stacks matrices by column, so it gives us the correct flattened
# vector.

# MOAI implements the complementarity as x1*x2 == 0. TODO: Change this to <= 0.
config = Dict(:ReLU => MOAI.ReLUQuadratic())

m = JuMP.Model()
JuMP.@variable(m, 0.0 <= x[1:height_dim, 1:length_dim] <= 1.0, start = 0.5)
y, _ = MOAI.add_predictor(m, predictor, vec(x); config)
JuMP.@constraint(m, 0.0 .<= y .<= 1.0)

# Minimize 1-norm of deviation from reference image using slack variables
JuMP.@variable(m, slack_pos[1:height_dim, 1:length_dim] >= 0.0)
JuMP.@variable(m, slack_neg[1:height_dim, 1:length_dim] >= 0.0)
JuMP.@constraint(m, x .- xref .== slack_pos .- slack_neg)
JuMP.@objective(m, Min, sum(slack_pos .+ slack_neg))

JuMP.@constraint(m, y[adversarial_target_index] >= threshold)

optimizer = JuMP.optimizer_with_attributes(
    Ipopt.Optimizer,
    "linear_solver" => "ma27",
)
JuMP.set_optimizer(m, optimizer)
JuMP.optimize!(m)

predicted_adversarial_output = JuMP.value.(y)
adversarial_x = JuMP.value.(x)

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

plot_image(adversarial_x)
