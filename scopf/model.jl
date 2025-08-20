import PowerModelsSecurityConstrained as PMSC
import PowerModels as PM
import InfrastructureModels as IM
import JuMP
import PythonCall
import MathOptAI as MOAI

SCOPF_DIR = dirname(@__FILE__)
include(joinpath(SCOPF_DIR, "nn-config.jl"))

"""Build a SCOPF model, possibly with a stability surrogate

Parameters
----------

data::Dict{String,Any}
    Data obtained by parsing input data files with
    PowerModelsSecurityConstrained.build_c1_pm_model.

stability_surrogate::MathOptAI.PytorchModel
    Default is nothing. We may allow other types if we use predictors from other
    sources.

stability_threshold::Float64
    Default is 59.4 (Hz, presumably)

surrogate_input_id::Int
    Controls how inputs and outputs of surrogate are managed.
    Default is 1, which corresponds to generator and load levels being used as
    inputs, and outputs corresponding to minimum frequency at each bus.

surrogate_params::Dict{Symbol,Any}
    Arguments sent to MathOptAI.add_predictor

"""
function build_scopf(
    data::Dict{String,Any};
    stability_surrogate::Union{Nothing,MOAI.PytorchModel} = nothing,
    stability_threshold::Float64 = 59.4,
    surrogate_input_id::Int = 1,
    surrogate_params::Dict{Symbol,Any} = Dict{Symbol,Any}(),
)
    _t = time()
    if !haskey(data, "gen_flow_cuts")
        data["gen_flow_cuts"] = []
    end
    if !haskey(data, "branch_flow_cuts")
        data["branch_flow_cuts"] = []
    end
    multinetwork = PMSC.build_c1_scopf_multinetwork(data)
    pm = IM.InitializeInfrastructureModel(
        PM.ACPPowerModel,
        multinetwork,
        PM._pm_global_keys,
        :pm,
        setting = Dict("output" => Dict("duals" => true)),
    )
    # This preprocesses data to get common lookups that are necessary to build models
    # (e.g. set of arcs).
    PM.ref_add_core!(pm.ref)
    # This adds challenge 1-specific data (only shunt data, I think)
    PMSC.ref_c1!(pm.ref, pm.data)

    # This populates pm with a JuMP model (pm.model).
    PMSC.build_c1_scopf(pm)
    # Other options that we're not currently using:
    #PMSC.build_c1_scopf_cuts(pm)
    #PMSC.build_c1_opf_shunt(pm)
    #PMSC.build_c1_scopf_cuts_soft(pm)

    t_pmsc = time() - _t

    if stability_surrogate !== nothing
        # PythonCall needs to be imported explicitly for us to use the MathOptAI
        # PythonCall extensions?
        inputs = INPUT_GETTER[surrogate_input_id](data, pm.model)

        # Outputs are min frequencies at each bus
        t_start = time()
        # NOTE: If we support other types of predictors, we may need
        # pop reduced_space and handle it separately.
        outputs, formulation = MOAI.add_predictor(
            pm.model,
            stability_surrogate,
            inputs;
            surrogate_params...,
        )
        t_predictor = time() - t_start
        println("TIME TO ADD PREDICTOR: $t_predictor")

        t_start = time()
        JuMP.@constraint(pm.model, outputs .>= stability_threshold)
        t_constraint = time() - t_start
        println("TIME TO ADD CONSTRAINT: $t_constraint")
    else
        t_predictor = 0.0
        t_constraint = 0.0
        outputs, formulation = nothing, nothing
    end
    info = (
        time = (
            scopf = t_pmsc,
            predictor = t_predictor,
            constraint = t_constraint,
        ),
        surrogate_outputs = outputs,
        surrogate_formulation = formulation,
    )
    return pm, info
end

# The contingency model is missing bounds on p, q, and vm. Imposing these bounds
# leads to a more restrictive version of the problem that should be feasible (according
# to the evaluator) if a solution is produced. However, it may be too restrictive
# to reliably find a feasible solution with Ipopt. Without the bounds, however, we
# may encounter solutions that are infeasible according to the evaluator.
# TODO: Add CLI option to apply these bounds
function apply_contingency_bounds(multinetwork, pm)
    for (key, nw) in multinetwork["nw"]
        # Basecase model already has these bounds
        if key != "0"
            cont_idx = key
            for (idx, gen) in nw["gen"]
                if gen["gen_status"] == 1
                    pg = JuMP.variable_by_name(pm.model, "$(cont_idx)_pg[$idx]")
                    qg = JuMP.variable_by_name(pm.model, "$(cont_idx)_qg[$idx]")
                    JuMP.@constraint(pm.model, gen["pmin"] <= pg <= gen["pmax"])
                    JuMP.@constraint(pm.model, gen["qmin"] <= qg <= gen["qmax"])
                end
            end
            for (idx, bus) in nw["bus"]
                vm = JuMP.variable_by_name(pm.model, "$(cont_idx)_vm[$idx]")
                # These bounds have not been applied in this model because voltage
                # magnitudes on generator buses have been fixed.
                JuMP.@constraint(pm.model, bus["evlo"] <= vm <= bus["evhi"])
            end
        end
    end
end

# Some of these arguments are redundant. TODO: Reduced number of arguments here.
function write_solution_files(
    network,
    pm,
    result,
    solfiles,
)
    # deepcopy so we can update the original network data with the basecase solution
    basecase_solution_network = deepcopy(network)
    PM.update_data!(basecase_solution_network, result["solution"]["nw"]["0"])
    # The correction step appears to be necessary in the base case. Without it, we get
    # a pgmax violation. This seems to be setting values to zero for offline devices,
    # which I guess aren't present in the solution data structure?
    PMSC.correct_c1_solution!(basecase_solution_network)
    PMSC.write_c1_solution1(basecase_solution_network, solution_file=solfiles[1])

    n_contingencies = length(network["branch_contingencies"]) + length(network["gen_contingencies"])
    # Is cont_order always consistent with the "network index" of the contingencies?
    contingency_solutions = [result["solution"]["nw"]["$conid"] for conid in 1:n_contingencies]
    cont_order = PMSC.contingency_order(network)
    for i in 1:n_contingencies
        # The solution2 writer assumes these fields exist
        contingency_solutions[i]["label"] = cont_order[i].label
        contingency_solutions[i]["cont_type"] = cont_order[i].type
        contingency_solutions[i]["cont_comp_id"] = cont_order[i].idx
        # This is present in some of Carleton's solution generation code, but it
        # doesn't appear to be necessary here.
        #contingency_solutions[i]["gen"]["$(cont_order[i].idx)"] = Dict("pg"=>0.0, "qg"=>0.0)

        # Work around a bug where shunts are only written to solution2 if they exist
        # in the contingency_solution data structure.
        contingency_solutions[i]["shunt"] = Dict(
            idx => Dict("gs" => shunt["gs"], "bs" => shunt["bs"])
            for (idx, shunt) in network["shunt"]
        )
    end

    # This step may be necessary at some point? update-data appears to only be necessary
    # if we are then going to try to correct the solutions.
    #for (idx, cont) in enumerate(cont_order)
    #    contingency_solution_network = multinetwork["nw"]["$idx"]
    #    PM.update_data!(contingency_solution_network, contingency_solutions[idx])
    #    PMSC.correct_c1_contingency_solution!(contingency_solution_network, contingency_solutions[idx])
    #end

    PMSC.write_c1_solution2(
        basecase_solution_network,
        contingency_solutions,
        solution_file=solfiles[2],
    )

    println("Wrote Solution 1 to $(solfiles[1])")
    println("Wrote Solution 2 to $(solfiles[2])")
end
