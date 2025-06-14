import JuMP
import MathOptAI as MOAI
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

function _collect_layers!(
    formulation::MOAI.AbstractFormulation,
    layers::Vector{<:MOAI.AbstractFormulation},
)
    push!(layers, formulation)
    return nothing
end

function _collect_layers!(
    formulation::MOAI.PipelineFormulation,
    layers::Vector{<:MOAI.AbstractFormulation},
)
    for layer in formulation.layers
        _collect_layers!(layer, layers)
    end
    return nothing
end

function get_layers(formulation::MOAI.AbstractFormulation)
    layers = MOAI.AbstractFormulation[]
    _collect_layers!(formulation, layers)
    return layers
end
