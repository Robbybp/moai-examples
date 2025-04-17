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

include("small-model.jl")
m, formulation = make_small_model()
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
# - Gets constraint types, iterates over them
# - Gets constraint indices of the given type from `backend`
# - get the constraint function for this index
# - index-map maps original index (in backend) to new index

#nlp_data = NLPModelsJuMP._nlp_block(moimodel)
#NLPModelsJuMP.parser_NL(nlp_data, hessian = true)

include("formulation.jl")
nn_vars, nn_cons = get_vars_cons(formulation)

include("nlpmodels.jl")
# These are the var/con orders in the MathOptNLPModel.
varorder, conorder = get_var_con_order(m)
ScalarFunction = Union{
    MOI.ScalarAffineFunction,
    MOI.ScalarQuadraticFunction,
    MOI.ScalarNonlinearFunction,
}
# If we have vector constraints, we have to do something fancier to get the coordinates
@assert all([MOI.get(m, MOI.ConstraintFunction(), c) isa ScalarFunction for c in conorder])
var_idx_map = Dict{JuMP.VariableRef,Int}(var => i for (i, var) in enumerate(varorder))
con_idx_map = Dict{JuMP.ConstraintRef,Int}(con => i for (i, con) in enumerate(conorder))

nn_vindices = [var_idx_map[v] for v in nn_vars]
nn_cindices = [con_idx_map[c] for c in nn_cons]

# Apply offset of nvar+nslack to get constraints' locations in the KKT system
nvar = length(varorder)
ind_cons = MadNLP.get_index_constraints(nlp)
nslack = length(ind_cons.ind_ineq)

# Get the coordinates of NN vars/cons in the lower-left Jacobian
# Columns (variables)
nn_kktcols = nn_vindices
# Rows (constraints) are offset by the size of the Hessian block
nn_kktrows = nn_cindices .+ (nvar + nslack)

cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)
linear_solver = SchurComplementSolver
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    ind_cons,
    linear_solver,
)
MadNLP.initialize!(kkt)
kkt_matrix = MadNLP.get_kkt(kkt)
submatrix = kkt_matrix[nn_kktrows, nn_kktcols]
