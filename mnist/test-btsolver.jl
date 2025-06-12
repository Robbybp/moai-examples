import JuMP
import LinearAlgebra
import MadNLP
import MadNLPHSL
import MathProgIncidence
import NLPModels
import SparseArrays
using Test
using Printf

include("btsolver.jl")
include("models.jl")
include("nlpmodels.jl")
include("linalg.jl")

function _test_matrix(csc::SparseArrays.SparseMatrixCSC; atol=1e-8)
    csc = SparseArrays.sparse(csc)
    dim = csc.m
    nrhs = 10
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)
    btsolver = BlockTriangularSolver(csc)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    baseline = csc \ rhs
    @test all(isapprox.(sol, baseline; atol))
end

function test_3x3_lt()
    matrix = [
        1.0 0.0 0.0;
        5.0 1.0 0.0;
        0.0 2.0 1.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    _test_matrix(matrix)
    return
end

"""
This test exercises the case where an unsymmetric column permutation matrix is used.
(I.e., if PAQ is our BT matrix, Q != Q^T.) This is important as, for symmetric
column permutations, we can reorder columns with the forward or inverse permutation
matrix without changing the result.
"""
function test_3x3_lt_unsym_perm()
    matrix = [
        0.0 0.0 1.0;
        1.0 0.0 5.0;
        2.0 1.0 0.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    _test_matrix(matrix)
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
    _test_matrix(matrix)
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
    _test_matrix(matrix)
    return
end

function test_nn_kkt()
    model, info = make_small_nn_model()
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    kkt_matrix = fill_upper_triangle(kkt_matrix)
    pivot_dim = length(pivot_indices)
    pivot_matrix = kkt_matrix[pivot_indices, pivot_indices]

    # Filter out constraint regularization nonzeros
    # By convention, constraints are the second half of the pivot indices
    to_ignore = Set(Int(pivot_dim / 2 + 1):pivot_dim)
    I, J, V = SparseArrays.findnz(pivot_matrix)
    to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    pivot_matrix = SparseArrays.sparse(I, J, V, pivot_dim, pivot_dim)

    _test_matrix(pivot_matrix)
end

@testset begin
    test_3x3_lt()
    test_3x3_lt_unsym_perm()
    test_4x4_blt()
    test_nn_jacobian()
    test_nn_kkt()
end
