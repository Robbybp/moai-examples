import JuMP
import MadNLP
import MadNLPHSL
import NLPModelsJuMP
import MathProgIncidence
import SparseArrays
import Profile

import DataFrames
import CSV

include("adversarial-image.jl")
include("linalg.jl")
include("formulation.jl")
include("nlpmodels.jl")
include("models.jl")

function precompile_linalg(; schur=true)
    model, info = make_tiny_model()
    _, _, kkt_matrix = get_kkt(model)
    pivot_indices = get_kkt_indices(model, info.variables, info.constraints)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    opt = SchurComplementOptions(; pivot_indices)
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
        println("Pivot subsolver   = $(opt.SchurSolver)")
    end
    rhs = ones(kkt_matrix.m)
    t_init_start = time()
    solver = Solver(kkt_matrix; opt)
    t_init = time() - t_init_start

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
    # TODO: Return something
    info = (;
        time = (;
            initialize = t_init,
            factorize = t_factorize,
            solve = t_solve,
        ),
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
    formname = reduced_space ? "reduced-space" : "full-space"
    println("Profiling the $formname formulation")
    # TODO: Don't rely on global data here.
    # But where to define the defaults?

    t_model_start = time()
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD;
        reduced_space = reduced_space,
    )
    nlp, kkt_system, kkt_matrix = get_kkt(model; Solver)
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    t_model = time() - t_model_start
    println("Time to build model and extract KKT: $t_model")
    rhs = ones(kkt_matrix.m)

    if schur
        opt = SchurComplementOptions(; ReducedSolver = Solver, PivotSolver = Solver, pivot_indices = pivot_indices)
        info = profile_solver(SchurComplementSolver, kkt_matrix; opt)
    else
        info = profile_solver(Solver, kkt_matrix)
    end

    newtimedata = merge((; model=t_model), info.time)
    result = (;
        time = newtimedata,
        nnz = info.nnz,
    )
    return result
end


if abspath(PROGRAM_FILE) == @__FILE__
    # TODO: CLI
    # Global data:
    IMAGE_INDEX = 7
    ADVERSARIAL_LABEL = 1
    THRESHOLD = 0.6

    PRECOMPILE = true
    if PRECOMPILE
        precompile_linalg()
    end

    nnfile = joinpath("nn-models", "mnist-relu128nodes4layers.pt")
    model, outputs, formulation = get_adversarial_model(
        nnfile, IMAGE_INDEX, ADVERSARIAL_LABEL, THRESHOLD
    )
    nlp, kkt_system, kkt_matrix = get_kkt(model)
    profile_solver(MadNLPHSL.Ma27Solver, kkt_matrix)
    profile_solver(MadNLPHSL.Ma27Solver, nnfile; reduced_space = true)

    PROFILE = false
    if PROFILE
        println("MadNLP.factorize!(::SchurComplementSolver) Profile:")
        Profile.print()
    end
end
