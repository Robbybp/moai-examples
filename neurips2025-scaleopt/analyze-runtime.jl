import JuMP
import MathOptInterface as MOI
import Ipopt
import DataFrames

include("model-getter.jl")

function get_ipopt_solve_time(model)
    t_solve = JuMP.solve_time(model)
    evaluator = model.moi_backend.optimizer.model.nlp_data.evaluator
    MOIExt = Base.get_extension(Ipopt, :IpoptMathOptInterfaceExt)
    if evaluator isa MOIExt._EmptyNLPEvaluator
        t_function = 0.0
        t_jacobian = 0.0
        t_hessian = 0.0
        println("NLP callback times")
        println("------------------")
        println("Empty NLP evaluator. We can't evaluate the callback times.")
    else
        t_function = evaluator.eval_objective_timer + evaluator.eval_constraint_timer
        t_jacobian = evaluator.eval_objective_gradient_timer + evaluator.eval_constraint_timer
        t_hessian = evaluator.eval_hessian_lagrangian_timer
        println("NLP callback times")
        println("------------------")
        println("Initialize:         $(evaluator.initialize_timer)")
        println("Objective:          $(evaluator.eval_objective_timer)")
        println("Grad. objective:    $(evaluator.eval_objective_gradient_timer)")
        println("Constraint          $(evaluator.eval_constraint_timer)")
        println("Constraint Jac:     $(evaluator.eval_constraint_jacobian_timer)")
        println("Lagrangian Hessian: $(evaluator.eval_hessian_lagrangian_timer)")
    end
    for (i, (f, s)) in enumerate(model.moi_backend.optimizer.model.vector_nonlinear_oracle_constraints)
        println("VNO $i function:    $(s.eval_f_timer)")
        println("VNO $i Jacobian:    $(s.eval_jacobian_timer)")
        println("VNO $i Hessian:     $(s.eval_hessian_lagrangian_timer)")
        t_function += s.eval_f_timer
        t_jacobian += s.eval_jacobian_timer
        t_hessian += s.eval_hessian_lagrangian_timer
    end
    println()
    return (;
        solve_total = t_solve,
        eval_function = t_function,
        eval_jacobian = t_jacobian,
        eval_hessian = t_hessian,
    )
end

function solve_model_with_ipopt(model)
    optimizer = JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => "ma57",
        "tol" => 1e-6,
        "print_user_options" => "yes",
        "print_timing_statistics" => "yes",
    )
    JuMP.set_optimizer(model, optimizer)
    JuMP.optimize!(model)
    t_solve = JuMP.solve_time(model)
    n_iterations = JuMP.barrier_iterations(model)
    success = JuMP.is_solved_and_feasible(model)
    # TODO: Combine these times with VNO timer
    objective_value = (success ? JuMP.objective_value(model) : NaN)
    timer = get_ipopt_solve_time(model)
    # TODO: Return solution
    return (;
        time = timer,
        n_iterations,
        success,
        objective_value,
    )
end

"""The following is code to set up and execute the parameter sweep.
I will probably want to rewrite this for each different sweep 
I want to perform.

This script contains the most basic parameter sweep highlighting
solve time and iteration count differences between full-space and
reduced-space (on CPU and GPU).
"""

# TODO: Add "cuda" if it is available
devices = Dict(:full_space => ["cpu"], :vector_nonlinear_oracle => ["cpu"])

model_names = ["mnist"]
# TODO: These NNs will depend on the model. We will need to look them up.
fnames = [
    "mnist-tanh128nodes4layers.pt",
    "mnist-tanh512nodes4layers.pt",
    #"mnist-tanh1024nodes4layers.pt",
    #"mnist-tanh2048nodes4layers.pt",
    #"mnist-tanh4096nodes4layers.pt",
    #"mnist-tanh8192nodes4layers.pt",
]
# In this experiment, the only "reduced-space" we care about is VNO.
formulations = [
    :full_space,
    :vector_nonlinear_oracle,
]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")
fpaths = map(f -> joinpath(nn_dir, f), fnames)

NSAMPLES = 2

#models = []
data = []
for model_name in model_names
    for fpath in fpaths
        for formulation in formulations
            for device in devices[formulation]
                for sample in 1:NSAMPLES
                    println("Model: $model_name")
                    println("NN: $fpath")
                    println("Formulation: $formulation")
                    println("Device: $device")
                    args = (; model = model_name, NN = basename(fpath), formulation, device, sample)
                    _t = time()
                    model = MODEL_GETTER[model_name](
                        fpath;
                        device = device,
                        sample_index = sample,
                        FORMULATION_TO_KWARGS[formulation]...,
                    )
                    t_model_build = time() - _t
                    println("Model build time: $t_model_build")

                    results = solve_model_with_ipopt(model)
                    time_info = merge(results.time, (; build_total = t_model_build))
                    # time_info.time will override results.time. This is what I want
                    info = merge(args, results, (; time = time_info))
                    #push!(models, model)
                    push!(data, info)
                end
            end
        end
    end
end
df = DataFrames.DataFrame(data)
println(df)
