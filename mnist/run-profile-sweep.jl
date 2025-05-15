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


SOLVER_TO_NAME = Dict(
    MadNLPHSL.Ma27Solver => "MA27",
    MadNLPHSL.Ma57Solver => "MA57",
    MadNLPHSL.Ma97Solver => "MA97",
)


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
            schur = params.schur,
        )
        # Insert 'nnz' field between params and times
        params = merge(params, (; nnz = info.nnz))
        params = (;
            Solver = SOLVER_TO_NAME[params.Solver],
            ID = _parse_nn_fname(params.fname),
            reduced = params.reduced,
            schur = params.schur,
        )
        push!(data, merge(params, info.time))
    end
    return data
end

function _parse_nn_fname(fname::String)
    fname = basename(fname)
    @assert length(split(fname, ".")) == 2
    fname = split(fname, ".")[1]
    @assert fname[1:length("mnist-")] == "mnist-"
    nchar = length(fname)
    activation_start = length("mnist-") + 1
    activation_end = activation_start
    while activation_end+1 <= nchar && !isdigit(fname[activation_end+1])
        activation_end += 1
    end
    activation_name = fname[activation_start:activation_end]

    nnode_start = activation_end + 1
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

NNFNAMES = [
    "mnist-relu128nodes4layers.pt",
    "mnist-relu128nodes8layers.pt",
    "mnist-relu256nodes4layers.pt",
    "mnist-relu256nodes8layers.pt",
    "mnist-relu512nodes4layers.pt",
    "mnist-relu512nodes8layers.pt",
    "mnist-relu768nodes4layers.pt",
    "mnist-relu1024nodes4layers.pt",
    "mnist-relu1024nodes8layers.pt",
    "mnist-relu1536nodes4layers.pt",
    "mnist-relu1536nodes8layers.pt",
    "mnist-relu1536nodes12layers.pt",
    "mnist-relu1792nodes8layers.pt",
    "mnist-relu2048nodes4layers.pt",
    "mnist-relu2048nodes5layers.pt",
    "mnist-relu2048nodes6layers.pt",
]

struct ProfileParams
    Solver::Type
    fname::String
    reduced::Bool
    schur::Bool
end
ProfileParams(Solver, fname, reduced; schur = false) = ProfileParams(Solver, fname, reduced, schur)

function Base.merge(p::ProfileParams, t::NamedTuple)
    params_t = (; Solver = p.Solver, fname = p.fname, reduced = p.reduced, schur = p.schur)
    return merge(params_t, t)
end

function _profile_nn_scaling()
    solver_types = [
        MadNLPHSL.Ma27Solver,
        MadNLPHSL.Ma57Solver, # TODO: How does MA57 do with Metis?
        MadNLPHSL.Ma97Solver,
    ]
    nnfnames = [
        "mnist-relu128nodes4layers.pt",
        "mnist-relu128nodes8layers.pt",
        "mnist-relu256nodes4layers.pt",
        "mnist-relu256nodes8layers.pt",
        "mnist-relu512nodes4layers.pt",
        "mnist-relu512nodes8layers.pt",
        #"mnist-relu768nodes4layers.pt",
        "mnist-relu1024nodes4layers.pt",
        "mnist-relu1024nodes8layers.pt",
        "mnist-relu1536nodes4layers.pt",
        "mnist-relu1536nodes8layers.pt",
        #"mnist-relu1536nodes12layers.pt",
        #"mnist-relu1792nodes8layers.pt",
        "mnist-relu2048nodes4layers.pt",
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
        ProfileParams(Solver, fname, reduced)
        for Solver in solver_types
        for fname in nnfnames
        for reduced in (false,)
    ]
    profile_params = [p for p in profile_params if !((p.Solver, p.fname) in excluded)]

    data = run_profile_sweep(; profile_params)
    df = DataFrames.DataFrame(data)
    display(df)
    return df
end

function _compare_full_reduced()
    solver_types = [
        MadNLPHSL.Ma27Solver,
        MadNLPHSL.Ma57Solver, # TODO: How does MA57 do with Metis?
        MadNLPHSL.Ma97Solver,
    ]
    nnfnames = [
        # IIRC, this is the largest network MA27 and MA97 work on...
        "mnist-relu512nodes4layers.pt",
        #"mnist-relu512nodes8layers.pt",
        "mnist-relu1024nodes4layers.pt",
    ]
    profile_params = [
        ProfileParams(Solver, fname, reduced)
        for reduced in (false, true)
        for fname in nnfnames
        for Solver in solver_types
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
    profile_params = [p for p in profile_params if !((p.Solver, p.fname) in excluded)]
    data = run_profile_sweep(; profile_params)
    df = DataFrames.DataFrame(data)
    display(df)
    return df
end

function _compare_schur()
    solver_types = [
        MadNLPHSL.Ma27Solver,
        MadNLPHSL.Ma57Solver, # TODO: How does MA57 do with Metis?
        MadNLPHSL.Ma97Solver,
    ]
    nnfnames = [
        # For testing
        #"mnist-relu128nodes4layers.pt",
        # IIRC, this is the largest network MA27 and MA97 work on...
        "mnist-relu512nodes4layers.pt",
        #"mnist-relu512nodes8layers.pt",
        "mnist-relu1024nodes4layers.pt",
    ]
    profile_params = [
        ProfileParams(Solver, fname, reduced, schur)
        for reduced in (false,)
        for schur in (false, true)
        for fname in nnfnames
        for Solver in solver_types
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
    profile_params = [p for p in profile_params if !((p.Solver, p.fname) in excluded)]
    data = run_profile_sweep(; profile_params)
    df = DataFrames.DataFrame(data)
    display(df)
    return df
end

#df = _profile_nn_scaling()
# The takeaway from this experiment is that symbolic factorization becomes
# the bottleneck at some point.
# And that, with about 2k node/layer, factorization takes about 12 s

#df = _compare_full_reduced()
# Takeaway: Reduced-space is fast.

df = _compare_schur()
# Takeaway: Schur is slow (although it can do symbolic factorization faster than
# MA27/97, which is not saying much). On the 1024-by-4 network, it is approximately
# 50x slower than it needs to be for symbolic and numeric factorization (e.g., 95 s vs 2 s
# for numeric).

# At some point I'll want to save the results, but for now I want to avoid too many
# out-of-sync results files clogging my directory.
WRITE = false
if WRITE
    open("profile.csv", "w") do io
        return CSV.write(io, df)
    end
end
