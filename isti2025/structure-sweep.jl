import DataFrames: DataFrame
import JuMP
import MathOptInterface as MOI
import Ipopt

function get_model_structure(model::JuMP.Model)
    nvar = length(JuMP.all_variables(model))
    cons = JuMP.all_constraints(model; include_variable_in_set_constraints = false)
    conobjs = JuMP.constraint_object.(cons)
    @assert all(map(c -> c.set isa MOI.AbstractScalarSet, conobjs))
    ncon = length(cons)
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
    #jnnz = length(Set(jacobian_structure))
    #hnnz = length(Set(hessian_structure))
    jnnz = length(jacobian_structure)
    hnnz = length(hessian_structure)
    return (; nvar, ncon, jnnz, hnnz)
end

include("../model-getter.jl")
include("nn-getter.jl")
include("localconfig.jl")
include("../pytorch.jl")
include("analyze-nns.jl")

tabledir = get_table_dir()

model_names = [
    "mnist",
    "scopf",
]
nn_dir = joinpath(dirname(dirname(@__FILE__)), "nn-models")
data = []
for model_name in model_names
    for fname in MODEL_TO_NNS[model_name]
        println("MODEL: $model_name")
        println("NN:    $fname")
        fpath = joinpath(nn_dir, fname)
        nninfo = analyze_nn(fpath)
        println(nninfo)
        model, formulation = MODEL_GETTER[model_name](fpath)
        res = get_model_structure(model)
        println(res)
        inputs = (; model=model_name, nn=fname)
        res = merge(inputs, nninfo, res)
        push!(data, res)
    end
end

df = DataFrame(data)
println(df)
fname = "structure.csv"
fpath = joinpath(tabledir, fname)
println("Writing results to $fpath")
CSV.write(fpath, df)
