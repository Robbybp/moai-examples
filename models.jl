import JuMP
import MathOptAI as MOAI

include("formulation.jl")

function update_kkt!(
    kkt::MadNLP.AbstractKKTSystem,
    nlp::NLPModels.AbstractNLPModel;
    x = nothing,
)
    # Need to update:
    # - Hessian
    # - Jacobian
    # - Regularization (set to zero? Or leave as default?)
    # - Σ_x, Σ_s (each for upper and lower bounds)
    # For now, I'd like to do the minimum necessary to give me a nonsingular KKT matrix
    hess_values = MadNLP.get_hessian(kkt)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    if x === nothing
        # FIXME: Some inertia tests fail when we use x0.
        #x = NLPModels.get_x0(nlp)
        x = ones(n)
    end
    λ = ones(m)

    # This is not always the case for some reason???
    # Due to fixed variables or something?
    #@assert NLPModels.get_nnzh(nlp) == length(hess_values)
    NLPModels.hess_coord!(nlp, x, λ, hess_values)

    jac_values = MadNLP.get_jacobian(kkt)
    @assert NLPModels.get_nnzj(nlp) == length(jac_values)
    NLPModels.jac_coord!(nlp, x, jac_values)

    # As far as I can tell, this is only used to set the primal
    # (and maybe dual?) diagonal.
    kkt.reg .= 0.0
    # These are unsafe wraps around KKT nonzeros. Here, I am initializing with
    # some combination of regularization and primal bound multipliers/slacks.
    # This is slightly different from what we would see in an IPM, but it's still
    # defensible. I'm just initializing bound multipliers to zero, then regularizing.
    #
    # Note that when I set `pr_diag` to zero, default MA57 gets way worse, but my
    # method doesn't change much. (Presumably this is due to numerical pivoting.)
    kkt.pr_diag .= 1.0
    kkt.du_diag .= 0.0
    return
end

function make_small_nn_model(;
    input_dim = 8,
    hidden_dim = 16,
    output_dim = 4,
    relaxation_parameter = 1e-6,
)
    m = JuMP.Model()
    JuMP.@variable(m, x[1:input_dim] >= 0)

    # TODO: Avoid random numbers here
    A1 = rand(hidden_dim, input_dim)
    b1 = rand(hidden_dim)
    A2 = rand(output_dim, hidden_dim)
    b2 = rand(output_dim)
    predictor = MOAI.Pipeline(
        MOAI.Affine(A1, b1),
        MOAI.ReLUQuadratic(; relaxation_parameter),
        #MOAI.Tanh(),
        MOAI.Affine(A2, b2),
        MOAI.SoftMax(),
    )
    # TODO: Initialize these variables so the NLP behaves better...
    y, formulation = MOAI.add_predictor(m, predictor, x)
    JuMP.@objective(m, Min, sum(x.^2) + sum(y.^2))
    variables, constraints = get_vars_cons(formulation)
    return m, (; formulation, variables, constraints)
end

# This doesn't really work. We tend to converge infeasible...
function get_small_nn_feasible_point(
    model::JuMP.Model;
    x = ones(length(model[:x])),
)
    JuMP.fix.(model[:x], x)
    JuMP.set_optimizer(model, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "linear_solver", "ma27")
    JuMP.optimize!(model)
    solution = Dict(x => JuMP.value(x) for x in JuMP.all_variables(model))
    return solution
end

function make_tiny_model()
    m = JuMP.Model()
    JuMP.@variable(m, x[1:3], start = 1.0)
    JuMP.@variable(m, y[1:2], start = 1.0)
    JuMP.set_lower_bound(x[1], 0.0)
    JuMP.set_lower_bound(x[2], 0.0)
    JuMP.set_lower_bound(x[3], -2.0)
    JuMP.set_lower_bound(y[1], 0.0)
    JuMP.set_lower_bound(y[2], 0.0)
    JuMP.set_upper_bound(x[1], 10.0)
    JuMP.set_upper_bound(y[1], 12.0)
    #JuMP.@constraint(m, eq1, x[1] + y[1] + x[2] == 10.0)
    #JuMP.@constraint(m, eq2, 2*x[2] + y[2] - x[3] == 12.0)
    #JuMP.@constraint(m, eq3, y[1] - y[2] == 3.0)
    #JuMP.@constraint(m, eq4, y[2] + 2*y[2] - x[3] == 7.0)
    JuMP.@constraint(m, eq1, x[1]^1.1 + y[1]^1.1 + x[2] == 10.0)
    JuMP.@constraint(m, eq2, 2*x[2] + y[2] - x[3] == 12.0)
    JuMP.@constraint(m, eq3, y[1]^1.1 - y[2] == 3.0)
    JuMP.@constraint(m, eq4, y[2]^1.1 + 2*y[2]^1.1 - x[3] == 7.0)
    JuMP.@constraint(m, ineq1, sum(x) + sum(y) <= 20.0)
    JuMP.@objective(m, Min, sum(x.^2) + sum(y.^2))
    variables = [y[1], y[2]]
    constraints = [eq3, eq4]
    return m, (; variables, constraints)
end
