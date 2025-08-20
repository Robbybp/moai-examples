"""This script currently runs a sweep and displays a DataFrame of structural
information about many models.

TODO:
- Put the sweep and printing in a separate file, leaving this file with only the
  core functionality

"""

# Just use whatever Python is on the Path. This gets around the fact that
# (a) I can't always use CondaPkg's conda installation (e.g., on HPC)
# and (b) I don't always have conda installed.
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import JuMP
import MathOptInterface as MOI
import Ipopt
import DataFrames
import CSV

include("localconfig.jl")
include("model-getter.jl")

function get_model_structure(model::JuMP.Model)
    nvar = length(JuMP.all_variables(model))
    cons = JuMP.all_constraints(model; include_variable_in_set_constraints = false)
    conobjs = map(JuMP.constraint_object, cons)
    # Make sure the only vector sets we have are _VectorNonlinearOracle
    @assert all(map(c -> c.set isa Ipopt._VectorNonlinearOracle || c.set isa MOI.AbstractScalarSet, conobjs))
    con_dims = map(
        # HACK: This is not general if I have other types of VectorSets.
        #
        # Really, we should be able to use the AbstractVectorSet.dimension field to get
        # dimension of any AbstractVectorSet. But VNO doesn't implement this.
        # Instead, it uses output_dimension.
        # Is this the right way to get the dimension of a general vector set?
        # The "dimension" of a constraint is not super well-defined...
        c -> c.set isa Ipopt._VectorNonlinearOracle ? c.set.output_dimension : 1,
        conobjs,
    )
    # Assume what we really mean by "N. constraints" is the number of rows in the
    # Jacobian...
    ncon = sum(con_dims)
    # Setting the optimizer is necessary to get nonzeros due to linear,
    # quadratic, and nonlinear parts of the model.
    JuMP.set_optimizer(model, Ipopt.Optimizer)
    # Not clear why I have to attach the optimizer...
    MOI.Utilities.attach_optimizer(model)
    ipopt = JuMP.unsafe_backend(model)
    MOIExt = Base.get_extension(Ipopt, :IpoptMathOptInterfaceExt)
    @assert MOIExt !== nothing
    MOIExt._setup_model(ipopt)
    hessian_structure = MOI.hessian_lagrangian_structure(ipopt)
    jacobian_structure = MOI.jacobian_structure(ipopt)
    jnnz = length(Set(jacobian_structure))
    hnnz = length(Set(hessian_structure))
    return (; nvar, ncon, jnnz, hnnz)
end

model_names = ["mnist", "scopf"]
formulations = [
    :full_space,
    # I can't even construct the reduced-space model for 128-by-4...
    #:reduced_space,
    :gray_box,
    :vector_nonlinear_oracle,
]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")

#models = []
data = []
for model_name in model_names
    for fname in MODEL_TO_NNS[model_name]
        fpath = joinpath(nn_dir, fname)
        for formulation in formulations
            println("Model: $model_name")
            println("NN: $fpath")
            println("Formulation: $formulation")
            args = (; model = model_name, NN = basename(fpath), formulation)
            model = MODEL_GETTER[model_name](fpath; FORMULATION_TO_KWARGS[formulation]...)
            structure_info = get_model_structure(model)
            info = merge(args, structure_info)
            #push!(models, model)
            push!(data, info)
        end
    end
end
df = DataFrames.DataFrame(data)
tabledir = get_table_dir()
fname = "structure.csv"
fpath = joinpath(tabledir, fname)
println("Writing results to $fpath")
CSV.write(fpath, df)
println(df)
