import JuMP
import MadNLP
import MadNLPHSL
import NLPModels
import SparseArrays
using Test

include("btsolver.jl")
include("models.jl")
include("nlpmodels.jl")

function test_3x3_lt()
    matrix = [
        1.0 0.0 0.0;
        5.0 1.0 0.0;
        0.0 2.0 1.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    dim = matrix.m
    nrhs = 10
    rhs = ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    for i in 1:nrhs
        @test all(isapprox.(sol[:,i], [1.0, -4.0, 9.0], atol=1e-8))
    end
    return
end

function test_4x4_blt()
    matrix = [
        0.0 1.0 0.0 0.0;
        1.0 5.0 0.0 3.0;
        0.0 0.0 4.0 2.0;
        2.0 0.0 0.0 2.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    dim = matrix.m
    nrhs = 10
    rhs = ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    correct_sol = [11/4, 1.0, 11/8, -9/4]
    for i in 1:nrhs
        @test all(isapprox.(sol[:,i], correct_sol, atol=1e-8))
    end
    return
end

function test_nn_jacobian()
    # NOTE: This model is constructed with random numbers.
    # TODO: Use a deterministic model here.
    model, info = make_small_nn_model()
    # ... do FixRef constraints not get picked up by NLPModels?
    #JuMP.fix.(model[:x], 2.0; force = true)
    JuMP.@constraint(model, model[:x] .== 2.0)
    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    vars, cons = get_var_con_order(model)
    x = [something(JuMP.start_value(var), 1.0) for var in vars]
    matrix = NLPModels.jac(nlp, x)
    display(matrix)

    dim = matrix.m
    nrhs = 10
    rhs = ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    display(sol)

    baseline = matrix \ rhs
    display(baseline)

    matches = isapprox.(sol, baseline, atol=1e-8)
    println("Discrepancies:")
    display(matches)
    nmatches = sum(matches)
    println("Number of matches: $nmatches")

    #@test all(isapprox.(sol, baseline, atol=1e-8))
    return
end

@testset begin
    test_3x3_lt()
    test_4x4_blt()
    test_nn_jacobian()
end
