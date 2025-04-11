import JuMP
import MadNLP
import MadNLPHSL
import MadNLPTests
import MathProgIncidence

include("adversarial-image.jl")

nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
image_index = 7
adversarial_label = 1
threshold = 0.6

m, outputs, formulation = get_adversarial_model(
    nnfile, image_index, adversarial_label, threshold
)

# My goal is to get the KKT matrix as SparseMatrixCSC
# Then I will:
# - Factorize it with MA27, backsolve, and compute inertia
# - Identify coordinates corresponding to NN output and internal variables
# - Perform a Schur complement decomposition WRT these coordinates, factorize,
#   backsolve, and solve the Schur system
# - Compare solutions and make sure they are the same

NLPModelsJuMP = MadNLPTests.NLPModelsJuMP

# Construct NLPModel from JuMP model
nlp = NLPModelsJuMP.MathOptNLPModel(m)

# Construct MadNLP's callback API from NLPModels' callback API
cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)

ind_cons = MadNLP.get_index_constraints(nlp)

linear_solver = MadNLPHSL.Ma27Solver

# MadNLP creates a KKT *system* object rather than just returning a KKT *matrix*
# I will re-implement this method/struct to contain additional coordinates to
# reduce.
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    ind_cons,
    linear_solver,
)

# Initialize KKT system with "default values". I assume these are primal variable
# start values and... some default values for duals...
MadNLP.initialize!(kkt)

# Don't need to call build_kkt! when we just initialized?
#MadNLP.build_kkt!(kkt)
#
# This returns a *reference* to the internal KKT matrix. That is, values will be
# updated automatically.
kkt_matrix = MadNLP.get_kkt(kkt)

# Factors will be stored inside the linear_solver instance.
MadNLP.factorize!(kkt.linear_solver)

@assert MadNLP.is_inertia(kkt.linear_solver)
# Inertia is in the format (pos, neg, zero)
inertia = MadNLP.inertia(kkt.linear_solver)

# Now I need to backsolve and get a solution that I can check later
b = ones(size(kkt_matrix)[1])
x = copy(b)
MadNLP.solve!(kkt.linear_solver, x)

# This is basically the workflow I need to go through.
# But how does this actually happen inside of MadNLP?
# - iterative refinement?
# - ReducedKKTSystem?
# Now I need to:
# - Implement this workflow with a Schur complement decomposition.
# - Figure out what methods I need to call to actually use a custom linear
#   solver during the solve.

function _collect_vars_cons!(
    formulation::MOAI.AbstractFormulation,
    predictor::MOAI.AbstractPredictor,
    variables::Vector{JuMP.VariableRef},
    constraints::Vector{<:JuMP.ConstraintRef},
)
    # By our convention, formulation contains intermediate variables and outputs
    append!(variables, formulation.variables)
    for con in formulation.constraints
        set = JuMP.MOI.get(con.model, JuMP.MOI.ConstraintSet(), con)
        # By default, we collect EqualTo constraints.
        if set isa JuMP.MOI.EqualTo
            push!(constraints, con)
        end
    end
    return nothing
end

function _collect_vars_cons!(
    formulation::MOAI.AbstractFormulation,
    predictor::MOAI.ReLUQuadratic,
    variables::Vector{JuMP.VariableRef},
    constraints::Vector{<:JuMP.ConstraintRef},
)
    append!(variables, formulation.variables)
    for con in formulation.constraints
        set = JuMP.MOI.get(con.model, JuMP.MOI.ConstraintSet(), con)
        fcn = JuMP.MOI.get(con.model, JuMP.MOI.ConstraintFunction(), con)
        # For ReLUQuadratic predictors, we include `y*x <= eps` constraints
        # as well as equality. The the KKT system on which we perform the Schur
        # complement, these will be: `y*x + s - eps == 0`
        if (
                set isa JuMP.MOI.EqualTo
                || (fcn isa JuMP.MOI.ScalarQuadraticFunction && set isa JuMP.MOI.LessThan)
        )
            push!(constraints, con)
        end
    end
    return nothing
end

function _collect_vars_cons!(
    formulation::MOAI.PipelineFormulation,
    predictor::MOAI.Pipeline,
    variables::Vector{JuMP.VariableRef},
    constraints::Vector{<:JuMP.ConstraintRef},
)
    for layer in formulation.layers
        _collect_vars_cons!(layer, layer.predictor, variables, constraints)
    end
    return nothing
end

function get_vars_cons(formulation::MOAI.AbstractFormulation)
    variables = JuMP.VariableRef[]
    constraints = JuMP.ConstraintRef[]
    _collect_vars_cons!(formulation, formulation.predictor, variables, constraints)
    return (; variables, constraints)
end

vars, cons = get_vars_cons(formulation)

# Not sure how I'm going to get these indices into the linear solver...
struct SchurComplementSolver <: MadNLP.AbstractLinearSolver
    inner_solver::MadNLP.AbstractLinearSolver
    var_indices::Vector{Int}
    con_indices::Vector{Int}
    # TODO: What else do I need? schur_solver to cache factors, inertia, etc.?
end
# linear_solver is constructed by KKTSystem?
# presumably I can use default_options? Or just pass this info in as
# options?

function SchurComplementSolver(
    csc::SparseArrays.SparseMatrixCSC,
)
end

opt = Dict("var_indices" => [1, 2, 3], "con_indices" => [4, 5, 6])
# This will be constructed as:
linear_solver = SchurComplementSolver(
    csc;
    opt = opt,
    logger = MadNLP.MadNLPLogger(),
)
