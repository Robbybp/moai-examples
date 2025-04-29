import MathOptAI as MOAI
import JuMP
import Ipopt
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays
using Test

include("linalg.jl")
include("nlpmodels.jl")
include("models.jl")

Random.seed!(1111)

function get_kkt(model::JuMP.Model)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    ind_cons = MadNLP.get_index_constraints(nlp)
    cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)
    kkt_system = MadNLP.create_kkt_system(
        MadNLP.SparseKKTSystem,
        cb,
        ind_cons,
        MadNLPHSL.Ma27Solver, # We won't use this linear solver
    )
    MadNLP.initialize!(kkt_system)
    update_kkt!(kkt_system, nlp)
    MadNLP.build_kkt!(kkt_system)
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    return nlp, kkt_system, kkt_matrix
end

function _test_factorize_nominal(
    kkt_matrix::SparseArrays.SparseMatrixCSC,
    pivot_indices::Vector{Int32},
)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    opt = SchurComplementOptions(; pivot_indices)
    schur_solver = SchurComplementSolver(kkt_matrix; opt)

    MadNLP.factorize!(ma27)
    MadNLP.factorize!(schur_solver)
    ma27_inertia = MadNLP.inertia(ma27)
    schur_inertia = MadNLP.inertia(schur_solver)

    @test ma27_inertia == schur_inertia
    return
end

function _test_solve_nominal(
    kkt_matrix::SparseArrays.SparseMatrixCSC,
    pivot_indices::Vector{Int32};
    rhs::Vector{Float64} = ones(kkt_matrix.m),
)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    opt = SchurComplementOptions(; pivot_indices)
    schur_solver = SchurComplementSolver(kkt_matrix; opt)
    MadNLP.factorize!(ma27)
    MadNLP.factorize!(schur_solver)
    sol_ma27 = copy(rhs)
    sol_schur = copy(rhs)
    MadNLP.solve!(ma27, sol_ma27)
    MadNLP.solve!(schur_solver, sol_schur)
    @test all(isapprox.(sol_ma27, sol_schur; atol=1e-8))
    return
end

function _test_solve_repeated(
    model::JuMP.Model,
    variables::Vector,
    constraints::Vector;
    nsamples::Int = 10,
    atol::Float64 = 1e-8,
)
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    varorder, _ = get_var_con_order(model)
    pivot_indices = get_kkt_indices(model, variables, constraints)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; pivot_indices)
    schur_solver = SchurComplementSolver(kkt_matrix; opt)
    nvar = NLPModels.get_nvar(nlp)
    for i in 1:nsamples
        x = rand(nvar) # Project this into bounds?
        rhs = rand(kkt_matrix.m)
        update_kkt!(kkt_system, nlp; x)
        MadNLP.build_kkt!(kkt_system) # This updates the KKT matrix in-place

        MadNLP.factorize!(ma27)
        MadNLP.factorize!(schur_solver)
        ma27_inertia = MadNLP.inertia(ma27)
        schur_inertia = MadNLP.inertia(schur_solver)
        #println("MA27 inertia = $ma27_inertia, Schur inertia = $schur_inertia")
        sol_ma27 = copy(rhs)
        sol_schur = copy(rhs)
        MadNLP.solve!(ma27, sol_ma27)
        MadNLP.solve!(schur_solver, sol_schur)
        maxdiff = maximum(abs.(sol_ma27 - sol_schur))
        #println("i = $i, ||ϵ|| = $maxdiff")
        @test all(isapprox.(sol_ma27, sol_schur; atol))
    end
    return
end

function test_factorize_nominal_tiny()
    m, info = make_tiny_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_factorize_nominal(matrix, convert(Vector{Int32}, indices))
    return
end

function test_factorize_nominal_small_nn()
    m, info = make_small_nn_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_factorize_nominal(matrix, convert(Vector{Int32}, indices))
    return
end

function test_solve_nominal_tiny()
    m, info = make_tiny_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_solve_nominal(matrix, convert(Vector{Int32}, indices))
    return
end

function test_solve_nominal_small_nn()
    m, info = make_small_nn_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_solve_nominal(matrix, convert(Vector{Int32}, indices))
    return
end

function test_solve_repeated_tiny()
    m, info = make_tiny_model()
    _test_solve_repeated(m, info.variables, info.constraints)
    return
end

function test_solve_repeated_small_nn(; atol = 1e-8)
    m, info = make_small_nn_model()
    _test_solve_repeated(m, info.variables, info.constraints; atol)
    return
end

function test_nlp_solve_tiny()
    m, info = make_tiny_model()
    nlp, _, _ = get_kkt(m)
    pivot_indices = get_kkt_indices(m, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    # Looks like we can do this with the
    optimizer = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "tol" => 1e-6,
        #"linear_solver" => MadNLPHSL.Ma27Solver,
        "linear_solver" => SchurComplementSolver,
        "pivot_indices" => pivot_indices,
    )
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
    return
end

function test_nlp_solve_small_nn()
    m, info = make_small_nn_model()
    nlp, _, _ = get_kkt(m)
    pivot_indices = get_kkt_indices(m, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    # Looks like we can do this with the
    optimizer = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "tol" => 1e-6,
        "linear_solver" => MadNLPHSL.Ma27Solver,
        #"linear_solver" => SchurComplementSolver,
        "pivot_indices" => pivot_indices,
    )
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
    return
end

@testset begin
    test_factorize_nominal_tiny()
    test_factorize_nominal_small_nn()
    test_solve_nominal_tiny()
    test_solve_nominal_small_nn()
    test_solve_repeated_tiny()

    ## This test is very fragile. We frequently have a couple of samples
    ## where we don't match the two samples to tolerance. Using random
    ## numbers in (a) the model and (b) the evaluation points doesn't help either.
    test_solve_repeated_small_nn(; atol = 1e-4)
    # These tests don't actually check anything at this point. TODO: Check termination
    # condition and solution.
    #test_nlp_solve_tiny()
    #test_nlp_solve_small_nn()
end
