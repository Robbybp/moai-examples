import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence
import SparseArrays
import Profile

import Printf
import DataFrames
import CSV

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("models.jl")
include("btsolver.jl")
include("kkt-partition.jl")

function precompile_linalg(; Solver=MadNLPHSL.Ma27Solver, schur=true)
    model, info = make_tiny_model()
    _, _, kkt_matrix = get_kkt(model; Solver)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = sort(pivot_indices)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; ReducedSolver=Solver, PivotSolver=Solver, pivot_indices)
    solver = SchurComplementSolver(kkt_matrix; opt)
    rhs = ones(kkt_matrix.m)
    MadNLP.factorize!(solver)
    MadNLP.solve!(solver, rhs)
    # TODO: Factorize with "baseline solver" in precompile section
end

if false
    model, outputs, formulation = get_adversarial_model(
        nnfile, image_index, adversarial_label, threshold
    )

    reduced_model, reduced_outputs, reduced_formulation = get_adversarial_model(
        nnfile, image_index, adversarial_label, threshold;
        reduced_space = true,
    )

    nlp, kkt_system, kkt_matrix = get_kkt(model)
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    #pivot_indices = sort(pivot_indices)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    rhs = ones(kkt_matrix.m)
    opt = SchurComplementOptions(; pivot_indices)
    solver = SchurComplementSolver(kkt_matrix; opt)

    display(kkt_matrix)

    NSAMPLES = 1
    for i in 1:NSAMPLES
        d = copy(rhs)
        # If we update values in the KKT system, we need to run the following:
        #update_kkt!(kkt_system, nlp; x)
        #MadNLP.build_kkt!(kkt_system)
        Profile.@profile MadNLP.factorize!(solver)
        MadNLP.solve!(solver, d)
    end
    println(solver.timer)
end

function profile_solver(
    Solver::Type,
    kkt_matrix::SparseArrays.SparseMatrixCSC;
    opt = MadNLP.default_options(Solver),
)
    println("Solver type:    $Solver")
    if Solver === SchurComplementSolver
        println("Reduced subsolver = $(opt.ReducedSolver)")
        println("Pivot subsolver   = $(opt.PivotSolver)")
    end
    rhs = ones(kkt_matrix.m)
    t_init_start = time()
    solver = Solver(kkt_matrix; opt)
    t_init = time() - t_init_start

    # For better or worse, we can't call introduce until solver is initialized.
    # This actually works well for Schur complement solvers because it means
    # I can include subsolvers in the string (of course, I could already do this
    # with multiple calls to `introduce`). But it's a little inconvenient here
    # (I need to wait for initialization to see confirmation that I'm using the
    # right solve).
    println(MadNLP.introduce(solver))
    println("-----------")
    println("initialize: $t_init")

    t_factorize_start = time()
    MadNLP.factorize!(solver)
    t_factorize = time() - t_factorize_start

    t_solve_start = time()
    MadNLP.solve!(solver, rhs)
    t_solve = time() - t_solve_start

    println("factorize:  $t_factorize")
    println("solve:      $t_solve")
    timer = (Solver === SchurComplementSolver) ? solver.timer : nothing
    info = (;
        time = (;
            initialize = t_init,
            factorize = t_factorize,
            solve = t_solve,
        ),
        # There is some redundant information here, but this is easier than adding
        # more fields to info.time.
        timer,
        nnz = SparseArrays.nnz(kkt_matrix),
    )
    return info
end

function profile_solver(
    Solver::Type,
    nnfile::String;
    reduced_space::Bool = false,
    schur::Bool = false,
    # TODO: Get default values for these if they don't exist.
    #image_index = nothing,
    #adversarial_label = nothing,
    #threshold = nothing,
)
    if schur && reduced_space
        error("schur=true && reduced_space=true doesn't make sense")
    end
    formname = reduced_space ? "reduced-space" : "full-space"
    println("Profiling the $formname formulation")
    # TODO: Don't rely on global data here.
    # But where to define the defaults?

    t_model_start = time()
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
        reduced_space = reduced_space,
    )
    println("Getting KKT matrix with solver type $Solver")
    nlp, kkt_system, kkt_matrix = get_kkt(model; Solver)
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    #pivot_indices = sort(pivot_indices)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    t_model = time() - t_model_start
    println("Time to build model and extract KKT: $t_model")
    rhs = ones(kkt_matrix.m)

    if schur || Solver == SchurComplementSolver
        # RECALL: It is critically important to use remap the indices of 
        # this partition into the correct sorted order.
        blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
        pivot_solver_opt = BlockTriangularOptions(; blocks)
        opt = SchurComplementOptions(;
            ReducedSolver = Solver,
            PivotSolver = BlockTriangularSolver,
            pivot_indices = pivot_indices,
            pivot_solver_opt,
        )
        info = profile_solver(SchurComplementSolver, kkt_matrix; opt)
    else
        info = profile_solver(Solver, kkt_matrix)
    end

    newtimedata = merge((; model=t_model), info.time)
    result = (;
        time = newtimedata,
        timer = info.timer,
        nnz = info.nnz,
    )
    return result
end


#if abspath(PROGRAM_FILE) == @__FILE__
    # TODO: CLI
    # Global data. Unfortunately, we rely on this in profile_solver(Solver, file)
    IMAGE_INDEX = 7
    ADVERSARIAL_LABEL = 1
    THRESHOLD = 0.6

    PRECOMPILE = true
    if PRECOMPILE
        precompile_linalg(; Solver=MadNLPHSL.Ma57Solver)
    end

    #nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
    #nnfile = joinpath("nn-models", "mnist-relu512nodes4layers.pt")
    #nnfile = joinpath("nn-models", "mnist-relu1024nodes4layers.pt")
    nnfile = joinpath("nn-models", "mnist-tanh1024nodes4layers.pt")
    #nnfile = joinpath("nn-models", "mnist-relu1536nodes4layers.pt")
    #nnfile = joinpath("nn-models", "mnist-relu2048nodes4layers.pt")
    #nnfile = joinpath("nn-models", "mnist-tanh2048nodes4layers.pt")
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
        reduced_space = false
    )
    nlp, kkt_system, kkt_matrix = get_kkt(model, Solver=MadNLPHSL.Ma57Solver)
    display(kkt_matrix)

    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    #pivot_indices = sort(pivot_indices)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    opt = SchurComplementOptions(;
        ReducedSolver = MadNLPHSL.Ma57Solver,
        PivotSolver = BlockTriangularSolver,
        pivot_indices = pivot_indices,
        pivot_solver_opt,
    )

    #@time results = profile_solver(MadNLPHSL.Ma57Solver, nnfile; schur = true)
    @time results = profile_solver(SchurComplementSolver, kkt_matrix; opt)
    println(results.timer)
    #results = profile_solver(MadNLPHSL.Ma57Solver, nnfile; schur = true)
    results = profile_solver(SchurComplementSolver, kkt_matrix; opt)
    println(results.timer)
    #@time results = profile_solver(MadNLPHSL.Ma57Solver, nnfile; schur = false)
    @time results = profile_solver(MadNLPHSL.Ma57Solver, kkt_matrix)
    println(results.timer)

    # The following is for examining specific submatrices in the Schur complement construction
    #pivot_vars, pivot_cons = get_vars_cons(formulation)
    #pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    #pivot_index_set = Set(pivot_indices)
    #@assert kkt_matrix.m == kkt_matrix.n
    #reduced_indices = filter(i -> !(i in pivot_index_set), 1:kkt_matrix.m)
    #P = pivot_indices
    #R = reduced_indices
    #B = kkt_matrix[P, R] + kkt_matrix[R, P]'
    ## These are columns that have at least one entry
    #nzcols = map(i -> B.colptr[i] < B.colptr[i+1], 1:length(R))
    #nnzcols = count(nzcols)
    #nnzcol_percent = 100.0 * nnzcols / length(R)
    #nnzcol_percent = Printf.@sprintf("%1.1f", nnzcol_percent)
    #println("$nnzcols out of $(length(R)) columns have entries ($(nnzcol_percent)%)")

    PROFILE = false
    if PROFILE
        println("MadNLP.factorize!(::SchurComplementSolver) Profile:")
        Profile.print()
    end
#end
