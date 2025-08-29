import SparseArrays
import Serialization
using Printf

import MadNLP
import NLPModelsJuMP

include("linalg.jl")
include("btsolver.jl")
include("model-getter.jl")
include("models.jl") # Necessary for update_kkt!...
include("nlpmodels.jl")

# TODO: Make this a function parameterized by model and NN
#model_name = "scopf"
model_name = "mnist"

#nnfname = "mnist-relu1024nodes4layers.pt"
#nnfname = "mnist-relu2048nodes4layers.pt"
nnfname = "mnist-tanh1024nodes4layers.pt"
#nnfname = "mnist-tanh2048nodes4layers.pt"

#nnfname = joinpath("scopf", "1000nodes7layers.pt")

nnfpath = joinpath("nn-models", nnfname)
_t = time()
model, formulation = MODEL_GETTER[model_name](nnfpath)
dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Make model")

Solver = SchurComplementSolver
#Solver = MadNLPHSL.Ma57Solver

# TODO: This should be a one-liner: model, formulation -> options
if Solver === SchurComplementSolver
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Get vars/cons from formulation")
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Get KKT indices")
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Convert to Int32")
    blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Partition by layer")
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    options = Dict{Symbol,Any}(
        :ReducedSolver => MadNLPHSL.Ma57Solver,
        :PivotSolver => BlockTriangularSolver,
        :pivot_indices => pivot_indices,
        :pivot_solver_opt => pivot_solver_opt,
    )
    opt_linear_solver = SchurComplementOptions(; options...)
    dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Get Schur options")
else
    options = Dict{Symbol,Any}()
    opt_linear_solver = MadNLP.default_options(Solver)
end

# Note that this involves instantiating a linear solver, so it will be very slow
# for large models, especially if we choose the wrong solver!
#nlp, kkt_system, kkt_matrix = get_kkt(model; Solver, opt_linear_solver)
# ^ This is actually unnecessary; all I need is the nlp
nlp = NLPModelsJuMP.MathOptNLPModel(model)
dt = time() - _t; println("[$(@sprintf("%1.2f", dt))] Get KKT matrix")

# This should reproduce exactly what we get in MadNLP.
# This does appear to give me the same performance profile I see in MadNLP.
# Note that the initialize! step is very important
madnlp = MadNLP.MadNLPSolver(
    nlp;
    tol = 1e-6,
    print_level = MadNLP.TRACE,
    max_iter = 0,
    linear_solver = Solver,
    options...,
)
#MadNLP.initialize!(madnlp)
stats = MadNLP.MadNLPExecutionStats(madnlp)
MadNLP.solve!(nlp, madnlp, stats)
kkt_system = madnlp.kkt
# Give us some primal regularization
# This doesn't really seem to help
#kkt_system.pr_diag .+= 1.0
# Transfer these values to the CSC matrix
#MadNLP.build_kkt!(kkt_system)
kkt_matrix = MadNLP.get_kkt(kkt_system)

display(kkt_matrix)
ave_nzmag = sum(abs.(kkt_matrix.nzval)) / length(kkt_matrix.nzval)
println("Average NZ magnitude: $ave_nzmag")

# We already have an initialized linear solver, but I'm going to initialize
# a new one to make sure I can time its initialization in isolation.

global _t = time()
linear_solver = Solver(kkt_matrix; opt = opt_linear_solver)
t_init = time() - _t; println("[$(@sprintf("%1.2f", t_init))] Initialize KKT System (and linear solver)")
println(MadNLP.introduce(linear_solver))

global _t = time()
MadNLP.factorize!(linear_solver)
t_factorize = time() - _t

# TODO: use a reasonable RHS by actually evaluating the constraints and
# Lagrangian?
#rhs = i .* Vector{Float64}(1:kkt_matrix.m)
#rhs = randn(kkt_matrix.m)
#
# I think `p` is the correct KKT RHS...
rhs = MadNLP.primal_dual(madnlp.p)
sol = copy(rhs)
global _t = time()
MadNLP.solve!(linear_solver, sol)
# At this point, I think I should just write my own iterative refinement...
#richardson_opt = MadNLP.RichardsonOptions(; richardson_max_iter = 20, richardson_tol = 1e-8, richardson_acceptable_tol = 1e-8)
#iterative_refiner = MadNLP.RichardsonIterator(
#    kkt_system;
#    opt = richardson_opt,
#    cnt = MadNLP.MadNLPCounters(start_time = time()),
#    logger = MadNLP.MadNLPLogger(MadNLP.TRACE, MadNLP.TRACE, nothing),
#)
#MadNLP.solve_refine!(madnlp.d, iterative_refiner, madnlp.p, madnlp._w4)
#sol .= MadNLP.primal_dual(madnlp.d)
#println(iterative_refiner.cnt)
t_solve = time() - _t

full_kkt = fill_upper_triangle(kkt_matrix)
abs_residual = maximum(abs.(full_kkt * sol - rhs))
println("residual = $abs_residual")

println("T. init.:     $t_init")
println("T. factorize: $t_factorize")
println("T. solve:     $t_solve")
println("Resid.:       $abs_residual")

# This has the advantage that, once I've extracted the matrices, I can
# extract whatever information I want. The downside is that I have to
# generate the matrices in another script, and this may be fairly
# solver-dependent or time-consuming.
if false    
    n_iter = 10

    dirname = "madnlp-matrices"
    fnames = map(i -> "iter$(@sprintf("%02d", i)).bin", 0:n_iter)
    fpaths = map(f -> joinpath(dirname, f), fnames)
    matrices = map(fp -> open(Serialization.deserialize, fp, "r"), fpaths)

    opt_fpath = joinpath(dirname, "opt.bin")
    opt = open(Serialization.deserialize, opt_fpath, "r")

    solver = SchurComplementSolver(matrices[1]; opt)
    for (i, matrix) in enumerate(matrices)
        local _t = time()
        solver.csc.nzval .= matrix.nzval
        # Using random entries makes things even worse.
        #solver.csc.nzval .= rand(SparseArrays.nnz(solver.csc))
        MadNLP.factorize!(solver)
        local rhs = i * Vector{Float64}(1:matrix.m)
        local sol = copy(rhs)
        MadNLP.solve!(solver, sol)
        local dt = time() - _t
        println("Iteration $i: $(@sprintf("%0.1f", dt)) s")
    end

    println(solver.timer)
end
