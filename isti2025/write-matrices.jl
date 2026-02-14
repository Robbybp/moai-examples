import MatrixMarket as MM

include("solve.jl")
include("../config.jl")
include("localconfig.jl")
include("../model-getter.jl")
include("nn-getter.jl")

model_names = [
    "mnist",
    "scopf",
    "lsv",
]
matrices = [
    "pivot",
    "schur",
    "original",
]

nn_lookup = Dict(
    "mnist" => "mnist-tanh128nodes4layers.pt",
    "scopf" => joinpath("scopf", "500nodes5layers.pt"),
    "lsv" => joinpath("lsv", "118_bus", "118_bus_32node.pt"),
)

#model_name = "scopf"
for model_name in model_names
    #nnfname = first(MODEL_TO_NNS[model_name])
    nnfname = nn_lookup[model_name]
    suffix = "-small"

    nnfpath = joinpath(get_nn_dir(), nnfname)
    model, formulation = MODEL_GETTER[model_name](nnfpath; sample_index = 1)
    pivot_vars, pivot_cons = get_vars_cons(formulation)
    pivot_indices = get_kkt_indices(model, pivot_vars, pivot_cons)
    pivot_indices = convert(Vector{Int32}, pivot_indices)

    varorder, conorder = get_var_con_order(model)
    pvarcon_set = Set(vcat(pivot_vars, pivot_cons))
    schur_vars = filter(v -> !(v in pvarcon_set), varorder)
    schur_cons = filter(c -> !(c in pvarcon_set), conorder)

    pivot_var_indices = get_kkt_indices(model, pivot_vars, [])
    pivot_con_indices = get_kkt_indices(model, [], pivot_cons)
    schur_var_indices = get_kkt_indices(model, schur_vars, [])
    schur_con_indices = get_kkt_indices(model, [], schur_cons)
    kktorder = vcat(
        schur_var_indices,
        schur_con_indices,
        pivot_var_indices,
        pivot_con_indices,
    )

    blocks = partition_indices_by_layer(model, formulation; indices = pivot_indices)
    pivot_solver_opt = BlockTriangularOptions(; blocks)
    madnlp_options = Dict{Symbol,Any}(
        #:kkt_system => MadNLP.ScaledSparseKKTSystem,
        :ReducedSolver => MadNLPHSL.Ma57Solver,
        :PivotSolver => BlockTriangularSolver,
        :pivot_indices => pivot_indices,
        :pivot_solver_opt => pivot_solver_opt,
    )

    nlp = NLPModelsJuMP.MathOptNLPModel(model)
    madnlp = MadNLP.MadNLPSolver(
        nlp;
        tol = 1e-6,
        #print_level = MadNLP.ERROR, #silent ? MadNLP.ERROR : MadNLP.TRACE,
        max_iter = 0,
        linear_solver = SchurComplementSolver,
        madnlp_options...,
    )
    MadNLP.initialize!(madnlp)

    for matrix_type in matrices
        println("Generating matrix `$matrix_type` for model `$model_name`")
        matrix, _ = MATRIX_GETTER[matrix_type](madnlp)
        if matrix_type == "original"
            matrix = matrix[kktorder, kktorder]
        end
        #order = order_lookup[matrix_type][
        #matrix = matrix[order]
        #dim = matrix.m
        #kkt_system = madnlp.kkt
        #rhs = MadNLP.primal_dual(madnlp.p)
        #rhs = rand(dim)
        #if model_name == "lsv" && matrix_type == "original"
        #    ma86 = MadNLPHSL.Ma86Solver(matrix)
        #    MadNLP.factorize!(ma86)
        #    inertia = MadNLP.inertia(ma86)
        #    println("Inertia = $inertia")
        #    sol = copy(rhs)
        #    MadNLP.solve!(ma86, sol)
        #    full_matrix = fill_upper_triangle(matrix)
        #    residual = maximum(abs.(full_matrix * sol - rhs))
        #    println("Residual = $residual")
        #end
        fname = join([model_name, matrix_type], "-") * "$suffix.mtx"
        fpath = joinpath("matrices", fname)
        MM.mmwrite(fpath, matrix)
        display(matrix)
        println("Wrote matrix to $fpath")
    end
end

#kkt, _ = MATRIX_GETTER["original"](madnlp)
#pivot, _ = MATRIX_GETTER["pivot"](madnlp)
#schur, _ = MATRIX_GETTER["schur"](madnlp)
#
#using UnicodePlots
#spy(kkt)
#spy(pivot)
#spy(schur)
