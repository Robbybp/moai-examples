# nlpmodels.jl and formulation.jl need to have been included.

function partition_indices_by_layer(
    model::JuMP.Model,
    formulation::MOAI.PipelineFormulation;
    indices::Union{Nothing,Vector} = nothing,
)
    # There is nothing special about the order of these variables/constraints.
    # They are simply those we want to pivot on.
    if indices === nothing
        variables, constraints = get_vars_cons(formulation)
        indices = get_kkt_indices(model, variables, constraints)
    end
    # This maps indices in the KKT matrix space to indices in the pivot space
    index_remap = Dict((p, i) for (i, p) in enumerate(indices))
    layers = get_layers(formulation)
    var_con_by_layer = [get_vars_cons(l) for l in layers]
    var_indices_by_layer = [get_kkt_indices(model, vars, []) for (vars, _) in var_con_by_layer]
    con_indices_by_layer = [get_kkt_indices(model, [], cons) for (_, cons) in var_con_by_layer]
    blocks = []
    for l in 1:length(layers)
        conindices = [index_remap[i] for i in con_indices_by_layer[l]]
        varindices = [index_remap[i] for i in var_indices_by_layer[l]]
        push!(blocks, (conindices, varindices))
    end
    for l in reverse(1:length(layers))
        conindices = [index_remap[i] for i in con_indices_by_layer[l]]
        varindices = [index_remap[i] for i in var_indices_by_layer[l]]
        push!(blocks, (varindices, conindices))
    end
    return blocks
end
