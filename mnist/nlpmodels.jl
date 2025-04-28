import JuMP
import MathOptInterface as MOI
import NLPModelsJuMP

_shape(::MOI.VariableIndex) = JuMP.ScalarShape()
_shape(::MOI.ScalarAffineFunction) = JuMP.ScalarShape()
_shape(::MOI.ScalarQuadraticFunction) = JuMP.ScalarShape()
_shape(::MOI.ScalarNonlinearFunction) = JuMP.ScalarShape()
_shape(::MOI.VectorAffineFunction) = JuMP.VectorShape()
_shape(::MOI.VectorQuadraticFunction) = JuMP.VectorShape()
_shape(::MOI.VectorNonlinearFunction) = JuMP.VectorShape()

function get_var_con_order(
    model::JuMP.Model
)::Tuple{Vector{JuMP.VariableRef}, Vector{JuMP.ConstraintRef}}
    moimodel = JuMP.backend(model)
    var_indices, con_indices = get_var_con_order(moimodel)
    vars = [JuMP.VariableRef(model, i) for i in var_indices]
    cons = Vector{JuMP.ConstraintRef}()
    for idx in con_indices
        fcn = MOI.get(moimodel, MOI.ConstraintFunction(), idx)
        con = JuMP.ConstraintRef(model, idx, _shape(fcn))
        push!(cons, con)
    end
    return vars, cons
end

function get_con_indices(model::MOI.ModelLike)
    linear = Vector{MOI.ConstraintIndex}()
    quadratic = Vector{MOI.ConstraintIndex}()
    nonlinear = Vector{MOI.ConstraintIndex}()
    contypes = MOI.get(model, MOI.ListOfConstraintTypesPresent())
    for (F, S) in contypes
        if F == MOI.VariableIndex
            continue
        end
        indices = MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
        for idx in indices
            fcn = MOI.get(model, MOI.ConstraintFunction(), idx)
            if fcn isa MOI.ScalarAffineFunction || fcn isa MOI.VectorAffineFunction
                push!(linear, idx)
            elseif fcn isa MOI.ScalarQuadraticFunction || fcn isa MOI.VectorQuadraticFunction
                push!(quadratic, idx)
            elseif fcn isa MOI.ScalarNonlinearFunction
                push!(nonlinear, idx)
            else
                error("Unsupported constraint function $F")
            end
        end
    end
    return (; linear, quadratic, nonlinear)
end

function get_var_con_order(
    model::MOI.ModelLike
)::Tuple{Vector{MOI.VariableIndex}, Vector{MOI.ConstraintIndex}}
    var_indices = MOI.get(model, MOI.ListOfVariableIndices())
    con_indices = get_con_indices(model)
    con_indices = vcat(con_indices...)
    return var_indices, con_indices
end

function get_kkt_indices(model::JuMP.Model, variables::Vector, constraints::Vector)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    varorder, conorder = get_var_con_order(model)
    var_idx_map = Dict(var => i for (i, var) in enumerate(varorder))
    con_idx_map = Dict(con => i for (i, con) in enumerate(conorder))
    vindices = [var_idx_map[v] for v in variables]
    cindices = [con_idx_map[c] for c in constraints]
    ind_cons = MadNLP.get_index_constraints(nlp)
    nvar = length(varorder)
    ncon = length(conorder)
    nslack = length(ind_cons.ind_ineq)
    kkt_dim = nvar + ncon + nslack
    kkt_vindices = vindices
    kkt_cindices = cindices .+ (nvar + nslack)
    kkt_indices = vcat(kkt_vindices, kkt_cindices)
    return kkt_indices
end
