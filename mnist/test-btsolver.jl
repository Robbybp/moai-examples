import SparseArrays
using Test

include("btsolver.jl")

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

@testset begin
    test_3x3_lt()
    test_4x4_blt()
end
