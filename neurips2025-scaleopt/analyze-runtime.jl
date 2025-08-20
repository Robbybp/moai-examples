ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import JuMP
import MathOptInterface as MOI
import Ipopt
import DataFrames
import CSV

include("localconfig.jl")
#include("model-getter.jl")
# ^ included by setup-compare-formulations

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
        t_solve_total = t_solve,
        t_eval_function = t_function,
        t_eval_jacobian = t_jacobian,
        t_eval_hessian = t_hessian,
    )
end

function solve_model_with_ipopt(model)
    optimizer = JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => "ma57",
        "tol" => 1e-6,
        "acceptable_tol" => 1e-5,
        "print_user_options" => "yes",
        "print_timing_statistics" => "yes",
    )
    JuMP.set_optimizer(model, optimizer)
    JuMP.optimize!(model)
    # TODO: NaN these if solve isn't successful?
    t_solve = JuMP.solve_time(model)
    n_iterations = JuMP.barrier_iterations(model)
    success = (
        JuMP.termination_status(model) in (JuMP.OPTIMAL, JuMP.ALMOST_OPTIMAL, JuMP.LOCALLY_SOLVED, JuMP.ALMOST_LOCALLY_SOLVED)
        && JuMP.primal_status(model) in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
    )
    # TODO: Combine these times with VNO timer
    objective_value = (success ? JuMP.objective_value(model) : NaN)
    timer = get_ipopt_solve_time(model)
    info = (; n_iterations, success, objective_value)
    info = merge(info, timer)
    # TODO: Return solution
    return info
end

"""The following is code to set up and execute the parameter sweep.
I will probably want to rewrite this for each different sweep 
I want to perform.

This script contains the most basic parameter sweep highlighting
solve time and iteration count differences between full-space and
reduced-space (on CPU and GPU).
"""

if abspath(PROGRAM_FILE) == @__FILE__

include("setup-compare-formulations.jl")

#models = []
data = []
for (index, model_name, fname, formulation, device, sample) in inputs
    # Note that nn_dir is defined in setup-compare-formulations.jl
    fpath = joinpath(nn_dir, fname)
    println("LOOP ELEMENT $index / $n_elements")
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
    t_build_total = time() - _t
    println("Model build time: $t_build_total")

    results = solve_model_with_ipopt(model)
    info = merge(args, results, (; t_build_total))
    #push!(models, model)
    push!(data, info)
end

df = DataFrames.DataFrame(data)
println(df)

tabledir = get_table_dir()
fname = "runtime-local.csv" # "local" as in "not distributed"... Maybe not the best name
fpath = joinpath(tabledir, fname)
println("Writing results to $fpath")
CSV.write(fpath, df)
println(df)

end
