import MathOptAI as MOAI
import JuMP
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import NLPModels, NLPModelsJuMP
import Random
Random.seed!(101)

input_dim = 8
hidden_dim = 16
output_dim = 4

optimizer = JuMP.optimizer_with_attributes(
    MadNLP.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => MadNLPHSL.Ma27Solver,
)

m = JuMP.Model()
#m = JuMP.direct_model(optimizer)
JuMP.@variable(m, x[1:input_dim] >= 0)

A1 = rand(hidden_dim, input_dim)
b1 = rand(hidden_dim)
A2 = rand(output_dim, hidden_dim)
b2 = rand(output_dim)
predictor = MOAI.Pipeline(
    MOAI.Affine(A1, b1),
    MOAI.ReLUQuadratic(; relaxation_parameter = 1e-6),
    MOAI.Affine(A2, b2),
    MOAI.SoftMax(),
)
y, formulation = MOAI.add_predictor(m, predictor, x)
JuMP.@objective(m, Min, sum(x.^2) + sum(y.^2))

include("linalg.jl")

solve = false
if solve
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
end

nlp = NLPModelsJuMP.MathOptNLPModel(m)
# To create this nlp, NLPModelsJuMP is running the parsers on backend(m)
moimodel = JuMP.backend(m)
# This index_map maps VariableIndex from backend to their position in
# ListOfVariableIndices
index_map, nvar, lvar, uvar, x0 = NLPModelsJuMP.parser_variables(moimodel)

# These extract the bound arrays, but don't tell us which JuMP constraints
# are at which positions.
# Implicitly, the order NLPModelsJuMP will use is (linear, quadratic, nonlinear).
# But what is the order within each of these?
# ^The order within linear/quadratic cons is actually specified in index_map
#  Why are the nonlinear constraints not specified here as well?
#
#nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon =
#  NLPModelsJuMP.parser_MOI(moimodel, index_map, nvar)
#nlp_data = NLPModelsJuMP._nlp_block(moimodel)
#nnln, nlcon, nl_lcon, nl_ucon = NLPModelsJuMP.parser_NL(nlp_data, hessian = true)

# This populates index_map with the correct indices
NLPModelsJuMP.parser_MOI(moimodel, index_map, nvar)
#nlp_data = NLPModelsJuMP._nlp_block(moimodel)
#NLPModelsJuMP.parser_NL(nlp_data, hessian = true)
