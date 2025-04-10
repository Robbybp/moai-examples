import Ipopt
import JuMP
import Plots
#ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import PythonCall
import MadNLP
import MadNLPHSL
import MadNLPMumps
import MathOptAI as MOAI
import MLDatasets

# TODO: make_model function
# - accepts filename, image index, adversary label, and formulation
# - returns JuMP model

# find_adversarial_image function
# - accepts the above, plus solver
# - solves the JuMP model, extracts the pixel values
# - returns a matrix of the pixel values, along with the network's predictions?

function get_adversarial_model(
    nnfile::String,
    image_index::Int,
    adversarial_label::Int,
    threshold::Float64,
)::JuMP.Model
    # Network is trained so that outputs represent 0-9
    adversarial_target_index = adversarial_label + 1
    predictor = MOAI.PytorchModel(nnfile)

    # TODO: Configurable data dir
    datadir = joinpath("data", "MNIST", "raw")
    test_data = MLDatasets.MNIST(; split = :test, dir = datadir)

    length_dim, height_dim = size(test_data[1].features)
    xref = test_data[image_index].features
    target_label = test_data[image_index].targets

    println("FINDING ADVERSARIAL EXAMPLE FOR IMAGE $(image_index)")
    println("LABEL FOR IMAGE $(image_index): $(target_label)")
    println("LOADING NN FROM FILE: $nnfile")
    torch = PythonCall.pyimport("torch")
    nn = torch.load(nnfile, weights_only = false)
    pyinput = torch.tensor(vec(xref))
    pyoutput = nn(pyinput).detach().numpy()
    output = PythonCall.pyconvert(Array, pyoutput)
    # Indices of `output` are 1-indexed, but predictions are 0-indexed
    prediction = argmax(output) - 1
    println("NN PREDICTS: $output")
    # How can I make sure my Julia and Python matrices are not transposes of each other?
    # This is really something that should be documented by the packages that convert the
    # images to arrays.
    #
    # In MLDatasets, images are indexed by width, then height. I.e. they are column-major.
    # Fortunately, `vec` stacks matrices by column, so it gives us the correct flattened
    # vector.

    config = Dict(:ReLU => MOAI.ReLUQuadratic())

    m = JuMP.Model()
    println
    #JuMP.@variable(m, 0.0 <= x[1:height_dim, 1:length_dim] <= 1.0, start = 0.5)
    JuMP.@variable(m, 0.0 <= x[i in 1:height_dim, j in 1:length_dim] <= 1.0, start = xref[i, j])
    y, _ = MOAI.add_predictor(m, predictor, vec(x); config)
    JuMP.@constraint(m, 0.0 .<= y .<= 1.0)

    # Minimize 1-norm of deviation from reference image using slack variables
    JuMP.@variable(m, slack_pos[1:height_dim, 1:length_dim] >= 0.0)
    JuMP.@variable(m, slack_neg[1:height_dim, 1:length_dim] >= 0.0)
    JuMP.@constraint(m, x .- xref .== slack_pos .- slack_neg)
    JuMP.@objective(m, Min, sum(slack_pos .+ slack_neg))

    JuMP.@constraint(m, y[adversarial_target_index] >= threshold)

    return m
end

function find_adversarial_image(
    nnfile::String,
    image_index::Int,
    adversarial_label::Int,
    threshold::Float64,
)
    m = get_adversarial_model(nnfile, image_index, adversarial_label, threshold)

    optimizer = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "linear_solver"=>MadNLPHSL.Ma27Solver,
        #"linear_solver"=>MadNLPMumps.MumpsSolver,
    )
    #optimizer = JuMP.optimizer_with_attributes(
    #    Ipopt.Optimizer,
    #    #"linear_solver" => "mumps",
    #    "print_user_options" => "yes",
    #    "print_timing_statistics" => "yes",
    #)
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)

    predicted_adversarial_output = JuMP.value.(y)
    adversarial_x = JuMP.value.(x)
    # TODO: return solve time, iteration count, etc.
    info = NamedTuple()
    return adversarial_x, info
end

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

if abspath(PROGRAM_FILE) == @__FILE__
    # TODO: Most of this goes in CLI, which will be in a separate script
    # - Why do I do CLI in a separage script?
    # - So I can reuse it?
    nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
    image_index = 7
    adversarial_label = 1
    threshold = 0.6

    find_adversarial_image(
        nnfile,
        image_index,
        adversarial_label,
        threshold,
    )
end
