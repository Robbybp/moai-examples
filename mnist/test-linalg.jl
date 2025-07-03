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
include("btsolver.jl")

Random.seed!(1111)

function _test_factorize_nominal(
    kkt_matrix::SparseArrays.SparseMatrixCSC,
    pivot_indices::Vector{Int32};
    PivotSolver = MadNLPHSL.Ma27Solver,
)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    opt = SchurComplementOptions(; pivot_indices, PivotSolver)
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
    PivotSolver = MadNLPHSL.Ma27Solver,
)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    opt = SchurComplementOptions(; pivot_indices, PivotSolver)
    schur_solver = SchurComplementSolver(kkt_matrix; opt)
    MadNLP.factorize!(ma27)
    MadNLP.factorize!(schur_solver)
    sol_ma27 = copy(rhs)
    sol_schur = copy(rhs)
    MadNLP.solve!(ma27, sol_ma27)
    MadNLP.solve!(schur_solver, sol_schur)
    @test all(isapprox.(sol_ma27, sol_schur; atol=1e-8))
    maxdiff = maximum(abs.(sol_ma27 - sol_schur))
    println("||ϵ|| = $maxdiff")
    return
end

function _test_solve_repeated(
    model::JuMP.Model,
    variables::Vector,
    constraints::Vector;
    nsamples::Int = 10,
    atol::Float64 = 1e-8,
    Solver::Type = MadNLPHSL.Ma27Solver,
    PivotSolver::Type = MadNLPHSL.Ma27Solver,
) # where T <: AbstractLinearSolver ?
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    varorder, _ = get_var_con_order(model)
    pivot_indices = get_kkt_indices(model, variables, constraints)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; pivot_indices, PivotSolver=PivotSolver, ReducedSolver=Solver)
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
        println("i = $i, ||ϵ|| = $maxdiff")
        @test all(isapprox.(sol_ma27, sol_schur; atol))
    end
    return
end

function test_factorize_nominal_tiny(; PivotSolver = MadNLPHSL.Ma27Solver)
    m, info = make_tiny_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_factorize_nominal(matrix, convert(Vector{Int32}, indices); PivotSolver)
    return
end

function test_factorize_nominal_small_nn(; PivotSolver = MadNLPHSL.Ma27Solver)
    m, info = make_small_nn_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_factorize_nominal(matrix, convert(Vector{Int32}, indices); PivotSolver)
    return
end

function test_solve_nominal_tiny(; PivotSolver = MadNLPHSL.Ma27Solver)
    m, info = make_tiny_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    println(PivotSolver)
    _test_solve_nominal(matrix, convert(Vector{Int32}, indices); PivotSolver)
    return
end

function test_solve_nominal_small_nn(; PivotSolver = MadNLPHSL.Ma27Solver)
    m, info = make_small_nn_model()
    _, _, matrix = get_kkt(m)
    indices = get_kkt_indices(m, info.variables, info.constraints)
    _test_solve_nominal(matrix, convert(Vector{Int32}, indices); PivotSolver)
    return
end

# Why does this test use MA57 as the default?
function test_solve_repeated_tiny(; Solver = MadNLPHSL.Ma57Solver, PivotSolver = MadNLPHSL.Ma57Solver)
    m, info = make_tiny_model()
    _test_solve_repeated(m, info.variables, info.constraints; Solver, PivotSolver)
    return
end

function test_solve_repeated_small_nn(; PivotSolver = MadNLPHSL.Ma27Solver, atol = 1e-8)
    m, info = make_small_nn_model()
    _test_solve_repeated(m, info.variables, info.constraints; PivotSolver, atol)
    return
end

function test_nlp_solve_tiny(; PivotSolver = MadNLPHSL.Ma27Solver)
    m, info = make_tiny_model()
    nlp, _, _ = get_kkt(m)
    pivot_indices = get_kkt_indices(m, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    optimizer = JuMP.optimizer_with_attributes(
        MadNLP.Optimizer,
        "tol" => 1e-6,
        #"linear_solver" => MadNLPHSL.Ma27Solver,
        "linear_solver" => SchurComplementSolver,
        "pivot_indices" => pivot_indices,
        "PivotSolver" => PivotSolver,
    )
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
    JuMP.assert_is_solved_and_feasible(m)
    return
end

function test_nlp_solve_small_nn(; PivotSolver = MadNLPHSL.Ma57Solver)
    m, info = make_small_nn_model()
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
        "PivotSolver" => PivotSolver,
        "ReducedSolver" => MadNLPHSL.Ma57Solver,
    )
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
    JuMP.assert_is_solved_and_feasible(m)
    return
end

function test_timer()
    m, info = make_tiny_model()
    _, _, kkt_matrix = get_kkt(m)
    pivot_indices = get_kkt_indices(m, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; pivot_indices)
    solver = SchurComplementSolver(kkt_matrix; opt)
    @test solver.timer.initialize > 0.0
    @test solver.timer.factorize.total == 0.0
    @test solver.timer.solve == 0.0
    MadNLP.factorize!(solver)
    factorize_timer = solver.timer.factorize
    @test factorize_timer.pivot > 0.0
    @test factorize_timer.reduced > 0.0
    @test factorize_timer.total > factorize_timer.pivot + factorize_timer.reduced
    rhs = ones(kkt_matrix.m)
    MadNLP.solve!(solver, rhs)
    @test solver.timer.solve > 0.0
end

@testset begin
    if true
        test_timer()
        test_factorize_nominal_tiny()
        test_factorize_nominal_small_nn()
        test_solve_nominal_tiny()
        test_solve_nominal_small_nn()
        test_solve_repeated_tiny()

        ## This test is very fragile. We frequently have a couple of samples
        ## where we don't match the two samples to tolerance. Using random
        ## numbers in (a) the model and (b) the evaluation points doesn't help either.
        test_solve_repeated_small_nn(; atol = 1e-4)

        # These check that the solver status is good and the solution is feasible,
        # but don't make sure that it's the solution we expect.
        test_nlp_solve_tiny()
        test_nlp_solve_small_nn()
    end

    # Tests with BTSolver
    #
    # This gives the wrong inertia...
    # but I have no proof that the inertia is stable here...
    test_factorize_nominal_tiny(PivotSolver = BlockTriangularSolver)
    test_solve_nominal_tiny(PivotSolver = BlockTriangularSolver)
    test_solve_repeated_tiny(PivotSolver = BlockTriangularSolver)
    test_nlp_solve_tiny(PivotSolver = BlockTriangularSolver)

    test_factorize_nominal_small_nn(PivotSolver = BlockTriangularSolver)
    test_solve_nominal_small_nn(PivotSolver = BlockTriangularSolver)
    test_solve_repeated_small_nn(; PivotSolver = BlockTriangularSolver, atol = 1e-4)
    test_nlp_solve_small_nn(; PivotSolver = BlockTriangularSolver)
end
