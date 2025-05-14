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


function run_profile_sweep(;
    profile_params,
)
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
    return data
end

function _parse_nn_fname(fname::String)
    fname = basename(fname)
    @assert length(split(fname, ".")) == 2
    fname = split(fname, ".")[1]
    @assert fname[1:6] == "mnist-"
    nchar = length(fname)
    activation_len = 0
    while activation_len+1 <= nchar && !isdigit(fname[activation_len+1])
        activation_len += 1
    end
    activation_name = fname[1:activation_len]

    nnode_start = activation_len + 1
    nnode_end = nnode_start
    @assert isdigit(fname[nnode_start])
    while nnode_end + 1 <= nchar && isdigit(fname[nnode_end+1])
        nnode_end += 1
    end
    nnode = parse(Int, fname[nnode_start:nnode_end])

    # Next comes "nodes"
    @assert fname[nnode_end+1:nnode_end+5] == "nodes"
    nlayer_start = nnode_end + length("nodes") + 1
    nlayer_end = nlayer_start
    @assert isdigit(fname[nlayer_start])
    while nlayer_end + 1 <= nchar && isdigit(fname[nlayer_end+1])
        nlayer_end += 1
    end
    nlayer = parse(Int, fname[nlayer_start:nlayer_end])

    # By convention, "layers" should come next
    @assert fname[nlayer_end+1:nlayer_end+length("layers")] == "layers"

    return (activation_name, nnode, nlayer)
end

function _profile_nn_scaling()
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
        #"mnist-relu512nodes8layers.pt",
        #"mnist-relu768nodes4layers.pt",
        #"mnist-relu1024nodes4layers.pt",
        #"mnist-relu1024nodes8layers.pt",
        #"mnist-relu1536nodes4layers.pt",
        "mnist-relu1536nodes8layers.pt",
        #"mnist-relu1536nodes12layers.pt",
        "mnist-relu1792nodes8layers.pt",
        #"mnist-relu2048nodes4layers.pt",
        #"mnist-relu2048nodes5layers.pt",
        #"mnist-relu2048nodes6layers.pt",
    ]
    excluded = Set()
    for fname in nnfnames
        _, nnode, nlayer = _parse_nn_fname(fname)
        # MA27 and MA97 have very bad scaling in symbolic factorization
        # that makes factorizing with these networks impractical.
        if nnode > 512 || (nnode == 512 && nlayer == 8)
            push!(excluded, (MadNLPHSL.Ma27Solver, fname))
            push!(excluded, (MadNLPHSL.Ma97Solver, fname))
        end
    end

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
    profile_params = [p for p in profile_params if !((p[1], p[2]) in excluded)]

    data = run_profile_sweep(; profile_params)
    df = DataFrames.DataFrame(data)
    display(df)
    return df
end

_profile_nn_scaling()

MAIN = false
if MAIN # By default, we'll run everything, I guess
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
        #"mnist-relu512nodes8layers.pt",
        #"mnist-relu768nodes4layers.pt",
        "mnist-relu1024nodes4layers.pt",
        "mnist-relu1024nodes8layers.pt",
        "mnist-relu2048nodes4layers.pt",
        "mnist-relu2048nodes8layers.pt",
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
    
    data = run_profile_sweep(; profile_params)
    df = DataFrames.DataFrame(data)
    display(df)
    open("profile.csv", "w") do io
        return CSV.write(io, df)
    end
end
