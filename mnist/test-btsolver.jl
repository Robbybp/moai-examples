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

function test_3x3_lt()
    matrix = [
        1.0 0.0 0.0;
        5.0 1.0 0.0;
        0.0 2.0 1.0;
    ]
    matrix = SparseArrays.sparse(matrix)
    dim = matrix.m
    nrhs = 10
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    display(rhs)
    baseline = matrix \ rhs
    @test all(isapprox.(sol, baseline, atol=1e-8))
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
    dim = matrix.m
    nrhs = 10
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    display(rhs)
    baseline = matrix \ rhs
    @test all(isapprox.(sol, baseline, atol=1e-8))
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
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    baseline = matrix \ rhs
    @test all(isapprox.(sol, baseline, atol=1e-8))
    #correct_sol = [11/4, 1.0, 11/8, -9/4]
    #for i in 1:nrhs
    #    @test all(isapprox.(sol[:,i], correct_sol, atol=1e-8))
    #end
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
    igraph = MathProgIncidence.IncidenceGraphInterface(matrix)
    blocks = MathProgIncidence.block_triangularize(igraph)
    cblocks = [b[1] for b in blocks]
    vblocks = [b[2] for b in blocks] 
    block_indices = [[i for _ in b[1]] for (i, b) in enumerate(blocks)]
    block_indices = vcat(block_indices...)
    corder = vcat(cblocks...)
    vorder = vcat(vblocks...)
    display(matrix[corder, vorder])

    dim = matrix.m
    nrhs = 10
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)
    btsolver = BlockTriangularSolver(matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)

    baseline = matrix \ rhs

    #diff = abs.(sol - baseline)
    #for i in 1:dim
    #    # Find positions in baseline where this solution value shows up
    #    correct_positions = findall(j -> abs(sol[i] - baseline[j]) <= 1e-8, 1:dim)
    #    @assert length(correct_positions) == 1
    #    correct_position = only(correct_positions)
    #    println(
    #        @sprintf("%3d", i)
    #        * @sprintf(" %+1.2E", sol[i])
    #        * @sprintf(" %+1.2E", baseline[i])
    #        * @sprintf(" %3.2f", diff[i])
    #        * @sprintf("\t\tblock %3d", block_indices[i])
    #        * @sprintf(", should be %3d", correct_position)
    #    )
    #end

    @test all(isapprox.(sol, baseline, atol=1e-8))
    return
end

@testset begin
    test_3x3_lt()
    test_3x3_lt_unsym_perm()
    test_4x4_blt()
    test_nn_jacobian()
end
