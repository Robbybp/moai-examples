import JuMP

function _get_inputs_pg_qg_qd(network::Dict, model::JuMP.Model)
    # The inputs for this model are as follows:
    # - 45 inputs for generation real (?) power (MW)
    # - 45 inputs for generation MVAR (reactive power)
    # - 27 inputs for load real power
    gens = [gen for gen in values(network["gen"])]
    gens_by_bus = sort(
        gens,
        by=gen -> gen["source_id"][2:3],
    )
    loads = [load for load in values(network["load"])]
    loads_by_bus = sort(
        loads,
        by=load -> load["source_id"][2:3],
    )

    basecase_pg_variables = [
        100.0 * ifelse(
            # Only try to access the variable if the generator is online.
            # (Otherwise the variable doesn't exist, and we use zero for the
            # predictor input.)
            gen["gen_status"] == 1,
            JuMP.variable_by_name(model, "0_pg[$(gen["index"])]"),
            0.0,
        )
        for gen in gens_by_bus
    ]
    basecase_qg_variables = [
        100.0 * ifelse(
            gen["gen_status"] == 1,
            JuMP.variable_by_name(model, "0_qg[$(gen["index"])]"),
            0.0,
        )
        for gen in gens_by_bus
    ]
    load_values = [100.0 * load["pd"] for load in loads_by_bus]

    inputs = [
        basecase_pg_variables..., 
        basecase_qg_variables...,
        load_values...,
    ]
    return inputs
end

function _get_inputs_pg_pd(network::Dict, model::JuMP.Model)
    gens = [gen for gen in values(network["gen"])]
    gens_by_bus = sort(
        gens,
        by=gen -> gen["source_id"][2:3],
    )
    loads = [load for load in values(network["load"])]
    loads_by_bus = sort(
        loads,
        by=load -> load["source_id"][2:3],
    )

    basecase_pg_variables = [
        100.0 * ifelse(
            # Only try to access the variable if the generator is online.
            # (Otherwise the variable doesn't exist, and we use zero for the
            # predictor input.)
            gen["gen_status"] == 1,
            JuMP.variable_by_name(model, "0_pg[$(gen["index"])]"),
            0.0,
        )
        for gen in gens_by_bus
    ]
    load_values_p = [100.0 * load["pd"] for load in loads_by_bus]

    basecase_pmax = [ifelse(gen["gen_status"] == 1, gen["pmax"], 0.0) for gen in gens_by_bus]
    basecase_pmin = [ifelse(gen["gen_status"] == 1, gen["pmin"], 0.0) for gen in gens_by_bus]

    #randomly set start values
    #for gen in gens_by_bus
    #    if gen["gen_status"] == 1
    #        JuMP.set_start_value(JuMP.variable_by_name(model, "0_pg[$(gen["index"])]"),rand(Uniform(basecase_pmin[gen["index"]],basecase_pmax[gen["index"]])))
    #    end
    #end

    # Infiltrator.@infiltrate

    inputs = [
        basecase_pg_variables..., 
        load_values_p...,
    ]
    return inputs
end

INPUT_GETTER = Dict(
    1 => _get_inputs_pg_qg_qd,
    4 => _get_inputs_pg_pd,
)
