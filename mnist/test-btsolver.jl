import JuMP
import LinearAlgebra
import MadNLP
import MadNLPHSL
import MathProgIncidence
import NLPModels
import SparseArrays
using Test
using Printf

include("adversarial-image.jl")
include("btsolver.jl")
include("models.jl")
include("nlpmodels.jl")
include("linalg.jl")

function _test_matrix(
    csc::SparseArrays.SparseMatrixCSC;
    atol=1e-8,
    nrhs = 10,
    baseline_solver = "umfpack",
    skiptest = false,
)
    dim = csc.m
    rowscaling = LinearAlgebra.diagm(convert(Vector{Float64}, 1:dim))
    rhs = rowscaling * ones(dim, nrhs)

    if true
        _t = time()
        btsolver = BlockTriangularSolver(csc)
        t_init = time() - _t
        _t = time()
        factorize!(btsolver)
        t_fact = time() - _t
        sol = copy(rhs)
        _t = time()
        solve!(btsolver, sol)
        t_solve = time() - _t
    else
        _t = time()
        solver = MadNLPHSL.Ma57Solver(SparseArrays.tril(csc))
        t_init = time() - _t
        _t = time()
        MadNLP.factorize!(solver)
        t_fact = time() - _t
        sol = copy(rhs)
        _t = time()
        MadNLP.solve!(solver, sol)
        t_solve = time() - _t
    end

    if baseline_solver == "umfpack"
        baseline = csc \ rhs
    elseif baseline_solver == "ma57"
        solver = MadNLPHSL.Ma57Solver(SparseArrays.tril(csc))
        MadNLP.factorize!(solver)
        baseline = copy(rhs)
        MadNLP.solve!(solver, baseline)
    else
        error("baseline_solver argument must be \"umfpack\" or \"ma57\"")
    end
    if !skiptest
        @test all(isapprox.(sol, baseline; atol))
    end
    if !(all(isapprox.(sol, baseline; atol)))
        diff = abs.(sol .- baseline)
        ndiff = count(diff[:, 1] .> atol)
        maxdiff = maximum(diff)
        println("Solution does not match baseline")
        println("Max error: $maxdiff")
        println("N. errors: $ndiff / $dim")
    end
    return (;
        time = (;
            initialize = t_init,
            factorize = t_fact,
            solve = t_solve,
        ),
    )
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

function test_nn_kkt_symmetric_inverse()
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

    # By using the identity matrix as the RHS, we recover the inverse.
    rhs = LinearAlgebra.diagm(ones(pivot_dim))
    btsolver = BlockTriangularSolver(pivot_matrix)
    factorize!(btsolver)
    sol = copy(rhs)
    solve!(btsolver, sol)
    baseline = pivot_matrix \ rhs
    @test all(isapprox.(sol, baseline; atol = 1e-8))

    symmetric_diffs = []
    for i in 1:pivot_dim
        for j in 1:(i-1)
            push!(symmetric_diffs, abs(sol[i,j] - sol[j, i]))
        end
    end
    @test all(symmetric_diffs .<= 1e-8)
end

function test_mnist_nn_kkt()
    IMAGE_INDEX = 7
    ADVERSARIAL_LABEL = 1
    THRESHOLD = 0.6
    #nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
    nnfile = joinpath("nn-models", "mnist-relu1024nodes4layers.pt")
    #nnfile = joinpath("nn-models", "mnist-relu2048nodes4layers.pt")
    if !isfile(nnfile)
        return
    end
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
        reduced_space = false
    )
    nlp, kkt_system, kkt_matrix = get_kkt(model, Solver=MadNLPHSL.Ma57Solver)

    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_index_set = Set(pivot_indices)
    @assert kkt_matrix.m == kkt_matrix.n
    reduced_indices = filter(i -> !(i in pivot_index_set), 1:kkt_matrix.m)
    pivot_dim = length(pivot_indices)
    @assert pivot_dim % 2 == 0

    P = pivot_indices
    R = reduced_indices
    C_orig = kkt_matrix[P, P]

    # Filter out constraint regularization nonzeros
    # By convention, constraints are the second half of the pivot indices
    to_ignore = Set(Int(pivot_dim / 2 + 1):pivot_dim)
    I, J, V = SparseArrays.findnz(C_orig)
    to_retain = filter(k -> !(I[k] in to_ignore && J[k] in to_ignore), 1:length(I))
    I = I[to_retain]
    J = J[to_retain]
    V = V[to_retain]
    C = SparseArrays.sparse(I, J, V, C_orig.m, C_orig.n)
    C_full = fill_upper_triangle(C)
    nrhs = 100
    # We skip the test as there is a significant amount of error for these relatively
    # large systems.
    res = _test_matrix(C_full; nrhs, atol = 1e-5, skiptest = true)
    println("Timing breakdown")
    println("----------------")
    println("Initialization: $(res.time.initialize)")
    println("Factorization:  $(res.time.factorize)")
    println("Solve (x$nrhs):  $(res.time.solve)")
end

@testset begin
    test_3x3_lt()
    test_3x3_lt_unsym_perm()
    test_4x4_blt()
    test_nn_jacobian()
    test_nn_kkt()
    test_nn_kkt_symmetric_inverse()
    test_mnist_nn_kkt()
end
