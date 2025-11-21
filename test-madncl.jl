import MadNCL
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
include("adversarial-image.jl")

nnfile = joinpath("nn-models", "mnist-tanh1024nodes4layers.pt")
image_index = 7
adversarial_label = 1
threshold = 0.6
model, y, formulation = get_adversarial_model(
    nnfile,
    image_index,
    adversarial_label,
    threshold;
    relaxation_parameter = 0.0,
)
nlp = NLPModelsJuMP.MathOptNLPModel(model)

variables, constraints = get_vars_cons(formulation)
pivot_indices = get_kkt_indices(model, variables, constraints)
blocks = partition_indices_by_layer(model, formulation)
#JuMP.set_optimizer_attribute(m, "pivot_indices", pivot_indices)
#JuMP.set_optimizer_attribute(m, "PivotSolver", MadNLPHSL.Ma57Solver)
#JuMP.set_optimizer_attribute(m, "PivotSolver", BlockTriangularSolver)
#JuMP.set_optimizer_attribute(m, "pivot_solver_opt", BlockTriangularOptions(; blocks))
#JuMP.set_optimizer_attribute(m, "ReducedSolver", MadNLPHSL.Ma57Solver)

res = MadNCL.madncl(
    nlp;
    #linear_solver = MadNLPHSL.Ma57Solver,
    kkt_system = MadNLP.SparseKKTSystem,
    linear_solver = SchurComplementSolver,
    pivot_indices,
    PivotSolver = BlockTriangularSolver,
    ReducedSolver = MadNLPHSL.Ma57Solver,
    pivot_solver_opt = BlockTriangularOptions(; blocks),
)
