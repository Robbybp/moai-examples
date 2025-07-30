import MathOptAI as MOAI
import JuMP
import Ipopt
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays
import InteractiveUtils

include("linalg.jl")
include("nlpmodels.jl")
include("models.jl")
include("btsolver.jl")
include("kkt-partition.jl")

m, info = make_small_nn_model()
_, _, matrix = get_kkt(m)
indices = get_kkt_indices(m, info.variables, info.constraints)
blocks = partition_indices_by_layer(m, info.formulation; indices)
pivot_solver_opt = BlockTriangularOptions(; blocks)

indices = convert(Vector{Int32}, indices)
opt = SchurComplementOptions(;
    pivot_indices = indices,
    PivotSolver = BlockTriangularSolver,
    pivot_solver_opt,
)
schur_solver = SchurComplementSolver(matrix; opt)
InteractiveUtils.@code_warntype MadNLP.factorize!(schur_solver)
