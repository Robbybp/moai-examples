import SparseArrays
import Serialization
using Printf

include("linalg.jl")
include("btsolver.jl")

n_iter = 10

dirname = "madnlp-matrices"
fnames = map(i -> "iter$(@sprintf("%02d", i)).bin", 0:n_iter)
fpaths = map(f -> joinpath(dirname, f), fnames)
matrices = map(fp -> open(Serialization.deserialize, fp, "r"), fpaths)

opt_fpath = joinpath(dirname, "opt.bin")
opt = open(Serialization.deserialize, opt_fpath, "r")

solver = SchurComplementSolver(matrices[1]; opt)
for (i, matrix) in enumerate(matrices)
    _t = time()
    solver.csc.nzval .= matrix.nzval
    # Using random entries makes things even worse.
    #solver.csc.nzval .= rand(SparseArrays.nnz(solver.csc))
    MadNLP.factorize!(solver)
    rhs = i * Vector{Float64}(1:matrix.m)
    sol = copy(rhs)
    MadNLP.solve!(solver, sol)
    dt = time() - _t
    println("Iteration $i: $(@sprintf("%0.1f", dt)) s")
end

println(solver.timer)
