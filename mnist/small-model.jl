import JuMP
import MathOptAI as MOAI

function make_small_model(; relaxation_parameter = 1e-6)
    m = JuMP.Model()
    JuMP.@variable(m, x[1:input_dim] >= 0)

    A1 = rand(hidden_dim, input_dim)
    b1 = rand(hidden_dim)
    A2 = rand(output_dim, hidden_dim)
    b2 = rand(output_dim)
    predictor = MOAI.Pipeline(
        MOAI.Affine(A1, b1),
        MOAI.ReLUQuadratic(; relaxation_parameter = relaxation_parameter),
        #MOAI.Tanh(),
        MOAI.Affine(A2, b2),
        MOAI.SoftMax(),
    )
    y, formulation = MOAI.add_predictor(m, predictor, x)
    JuMP.@objective(m, Min, sum(x.^2) + sum(y.^2))
    return m, formulation
end
