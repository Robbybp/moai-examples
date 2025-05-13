"""Script to profile linear algebra for many combinations of solver,
formulation, and NN model.
"""

import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence
import SparseArrays
import Profile

import DataFrames
import CSV

#include("adversarial-image.jl")
#include("linalg.jl")
#include("formulation.jl")
#include("nlpmodels.jl")
#include("models.jl")

include("profile-linalg.jl")

# TODO: Precompile with different linear solvers
precompile_linalg()

# TODO: CLI
# Global data:
IMAGE_INDEX = 7
ADVERSARIAL_LABEL = 1
THRESHOLD = 0.6

solver_types = [
    #MadNLPHSL.Ma27Solver,
    MadNLPHSL.Ma57Solver, # TODO: How does MA57 do with Metis?
    #MadNLPHSL.Ma97Solver,
]
nnfnames = [
    #"mnist-relu128nodes4layers.pt",
    #"mnist-relu128nodes8layers.pt",
    #"mnist-relu256nodes4layers.pt",
    #"mnist-relu256nodes8layers.pt",
    #"mnist-relu512nodes4layers.pt",
    "mnist-relu512nodes8layers.pt",
    "mnist-relu768nodes4layers.pt",
    "mnist-relu1024nodes4layers.pt",
    "mnist-relu1024nodes8layers.pt",
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
