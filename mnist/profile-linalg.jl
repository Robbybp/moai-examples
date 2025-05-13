import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence
import SparseArrays
import Profile

import DataFrames
import CSV

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("models.jl")

PRECOMPILE = true
if PRECOMPILE
    model, info = make_tiny_model()
    _, _, kkt_matrix = get_kkt(model)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; pivot_indices)
    solver = SchurComplementSolver(kkt_matrix; opt)
    rhs = ones(kkt_matrix.m)
    MadNLP.factorize!(solver)
    MadNLP.solve!(solver, rhs)
    # TODO: Factorize with "baseline solver" in precompile section
end

# TODO: CLI
#
# Global data:
IMAGE_INDEX = 7
ADVERSARIAL_LABEL = 1
THRESHOLD = 0.6

#nnfile = joinpath("nn-models", "mnist-relu512nodes4layers.pt")
    
if false
    model, outputs, formulation = get_adversarial_model(
        nnfile, image_index, adversarial_label, threshold
    )
    
    reduced_model, reduced_outputs, reduced_formulation = get_adversarial_model(
        nnfile, image_index, adversarial_label, threshold;
        reduced_space = true,
    )
    
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    rhs = ones(kkt_matrix.m)
    opt = SchurComplementOptions(; pivot_indices)
    solver = SchurComplementSolver(kkt_matrix; opt)
    
    display(kkt_matrix)
    
    NSAMPLES = 1
    for i in 1:NSAMPLES
        d = copy(rhs)
        # If we update values in the KKT system, we need to run the following:
        #update_kkt!(kkt_system, nlp; x)
        #MadNLP.build_kkt!(kkt_system)
        Profile.@profile MadNLP.factorize!(solver)
        MadNLP.solve!(solver, d)
    end
    println(solver.timer)
end

function profile_solver(
    Solver::Type,
    kkt_matrix::SparseArrays.SparseMatrixCSC,
)
    rhs = ones(kkt_matrix.m)
    t_init_start = time()
    solver = Solver(kkt_matrix)
    t_init = time() - t_init_start

    t_factorize_start = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - t_factorize_start

    t_solve_start = time()
    MadNLP.solve!(solver, rhs)
    t_solve = time() - t_solve_start

    println(MadNLP.introduce(solver))
    println("-----------")
    println("initialize: $t_init")
    println("factorize:  $t_factorize")
    println("solve:      $t_solve")
    # TODO: Return something
    info = (;
        time = (;
            initialize = t_init,
            factorize = t_factorize,
            solve = t_solve,
        ),
        nnz = SparseArrays.nnz(kkt_matrix),
    )
    return info
end

function profile_solver(
    Solver::Type,
    nnfile::String;
    reduced_space::Bool = false,
)
    formname = reduced_space ? "reduced-space" : "full-space"
    println("Profiling the $formname formulation")
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
        reduced_space = reduced_space,
    )
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    rhs = ones(kkt_matrix.m)
    return profile_solver(Solver, kkt_matrix)
end

#profile_solver(MadNLPHSL.Ma27Solver, kkt_matrix)
#profile_solver(MadNLPHSL.Ma27Solver; reduced_space = true)

solver_types = [
    MadNLPHSL.Ma27Solver,
    MadNLPHSL.Ma57Solver,
    MadNLPHSL.Ma97Solver,
]
nnfnames = [
    "mnist-relu128nodes4layers.pt",
    "mnist-relu128nodes8layers.pt",
    "mnist-relu256nodes4layers.pt",
    "mnist-relu256nodes8layers.pt",
    "mnist-relu512nodes4layers.pt",
    "mnist-relu512nodes8layers.pt",
    "mnist-relu1024nodes4layers.pt",
]
excluded = Set([
    # MA27 and MA97 have very bad scaling in symbolic factorization
    # that makes factorizing with these networks impractical.
    (MadNLPHSL.Ma27Solver, "mnist-relu512nodes8layers.pt"),
    (MadNLPHSL.Ma27Solver, "mnist-relu1024nodes4layers.pt"),
    (MadNLPHSL.Ma97Solver, "mnist-relu512nodes8layers.pt"),
    (MadNLPHSL.Ma97Solver, "mnist-relu1024nodes4layers.pt"),
])

profile_params = [
    (;
        Solver,
        fname,
        reduced,
    )
    for Solver in solver_types
    for fname in nnfnames
    for reduced in (false,)
]

# Filter out combinations that we know aren't worth doing
# (e.g., because they take too long and we already know the bottleneck)
profile_params = [p for p in profile_params if !((p[1], p[2]) in excluded)]

# TODO: Move this to a CLI file
data = []
for params in profile_params
    println()
    println("Profiling with following parameters:")
    display(params)
    fpath = joinpath("nn-models", params.fname)
    info = profile_solver(
        params.Solver,
        fpath;
        reduced_space = params.reduced,
    )
    # Insert 'nnz' field between params and times
    params = merge(params, (; nnz = info.nnz))
    push!(data, merge(params, info.time))
end
df = DataFrames.DataFrame(data)
display(df)
open("profile.csv", "w") do io
    return CSV.write(io, df)
end

PROFILE = false
if PROFILE
    println("MadNLP.factorize!(::SchurComplementSolver) Profile:")
    Profile.print()
end
