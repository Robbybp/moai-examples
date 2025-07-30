import LinearAlgebra
using Test

include("blockdiagonal.jl")

function _test_matrix(matrix::Matrix; atol = 1e-8, nrhs = 10)
    dim, dim2 = matrix.size
    @assert dim == dim2
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)
    rowcc, colcc = connected_components(matrix)
    bd = BlockDiagonalView(matrix, rowcc, colcc)
    lu = LinearAlgebra.lu(bd)
    sol = copy(rhs)
    LinearAlgebra.ldiv!(lu, sol)
    baseline_sol = matrix \ rhs
    @test all(isapprox.(sol, baseline_sol; atol))
    return bd, lu, sol
end

function test_3x3_diagonal()
    matrix = [
        0.0 1.0 0.0;
        3.0 0.0 0.0;
        0.0 0.0 2.0;
    ]
    bd, _, _ = _test_matrix(matrix)
    # Make sure changes are applied directly to BlockDiagonalView's matrix
    @test matrix === bd.matrix
    matrix[1, 2] = 8.0
    @test bd.matrix[1, 2] == 8.0
    # Solve with the new matrix
    rhs = [3.0, 4.0, 5.0]
    baseline = matrix \ rhs
    lu = LinearAlgebra.lu(bd)
    sol = copy(rhs)
    LinearAlgebra.ldiv!(lu, sol)
    @test all(isapprox.(sol, baseline; atol = 1e-8))
    return
end

function test_4x4_blockdiagonal()
    matrix = [
        0.0 2.0 0.0 3.0;
        1.0 0.0 1.1 0.0;
        1.4 0.0 4.1 0.0;
        0.0 4.0 0.0 2.0;
    ]
    bd, lu, sol = _test_matrix(matrix)
    return
end

@testset "block-diagonal" begin
    test_3x3_diagonal()
    test_4x4_blockdiagonal()
end
