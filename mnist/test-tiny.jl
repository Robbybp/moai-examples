import MathOptAI as MOAI
import JuMP
import Ipopt
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays

include("linalg.jl")
include("nlpmodels.jl")
include("models.jl")

#function main()
    m = make_tiny_model()

    optimize = false
    if optimize
        JuMP.set_optimizer(m, Ipopt.Optimizer)
        JuMP.set_optimizer_attribute(m, "linear_solver", "ma27")
        JuMP.optimize!(m)
        println("x: $(JuMP.value.(m[:x]))")
        println("y: $(JuMP.value.(m[:y]))")
    end

    pivot_vars = [m[:y][1], m[:y][2]]
    pivot_cons = [m[:eq3], m[:eq4]]

    nlp = NLPModelsJuMP.MathOptNLPModel(m)
    varorder, conorder = get_var_con_order(m)
    var_idx_map = Dict(var => i for (i, var) in enumerate(varorder))
    con_idx_map = Dict(con => i for (i, con) in enumerate(conorder))
    vindices = [var_idx_map[v] for v in pivot_vars]
    cindices = [con_idx_map[c] for c in pivot_cons]
    nvar = length(varorder)
    ncon = length(conorder)
    ind_cons = MadNLP.get_index_constraints(nlp)
    nslack = length(ind_cons.ind_ineq)
    kkt_dim = nvar + ncon + nslack

    println("nvar   = $nvar")
    println("ncon   = $ncon")
    println("nslack = $nslack")

    pivot_vindices = vindices
    # Apply offset used in KKT matrix
    pivot_cindices = cindices .+ (nvar + nslack)
    pivot_indices = vcat(pivot_vindices, pivot_cindices)

    cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)
    kkt_system = MadNLP.create_kkt_system(
        MadNLP.SparseKKTSystem,
        cb,
        ind_cons,
        MadNLPHSL.Ma27Solver,
    )
    MadNLP.initialize!(kkt_system)
    update_kkt!(kkt_system, nlp)
    MadNLP.build_kkt!(kkt_system)
    kkt_matrix = MadNLP.get_kkt(kkt_system)
    display(kkt_matrix)

    println("Constructing SchurComplementSolver with pivot indices:")
    println("$pivot_indices")
    pivot_indices = convert(Vector{Int32}, pivot_indices)
    pivot_index_set = Set(pivot_indices)
    reduced_indices = filter(x -> !(x in pivot_index_set), 1:kkt_dim)
    opt = SchurComplementOptions(; pivot_indices = pivot_indices)
    schur_solver = SchurComplementSolver(kkt_matrix; opt)
    ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)

    # Have the KKT matrix. Now I can start testing some things...
    MadNLP.factorize!(schur_solver)
    MadNLP.factorize!(ma27)

    schur_inertia = MadNLP.inertia(schur_solver)
    ma27_inertia = MadNLP.inertia(ma27)
    println("Schur inertia: $schur_inertia")
    println("MA27 inertia:  $ma27_inertia")

    rhs = ones(kkt_dim)
    d_ma27 = copy(rhs)
    d_schur = copy(rhs)
    MadNLP.solve!(schur_solver, d_schur)
    MadNLP.solve!(ma27, d_ma27)

    println("d_schur")
    println(d_schur)
    println("d_ma27")
    println(d_ma27)

    println("d_schur[reduced_indices]")
    println(d_schur[reduced_indices])
    println("d_ma27[reduced_indices]")
    println(d_ma27[reduced_indices])

    display(schur_solver.csc)
    Random.seed!(513)
    for i in 1:10
        global rhs = rand(kkt_dim)
        x = rand(nvar)
        update_kkt!(kkt_system, nlp; x)
        # I think this is necessary...
        MadNLP.build_kkt!(kkt_system)
        #kkt_matrix = MadNLP.get_kkt_matrix(kkt_system)

        #display(schur_solver.csc)
        ## Reduced solver's matrix has not been updated
        #display(schur_solver.reduced_solver.csc)
        #display(schur_solver.schur_solver.csc)

        # We should be able to factorize and solve with the same linear solvers
        global d = copy(rhs)
        global d_ma27 = copy(rhs)
        MadNLP.factorize!(schur_solver)
        MadNLP.factorize!(ma27)
        global schur_inertia = MadNLP.inertia(schur_solver)
        global ma27_inertia = MadNLP.inertia(ma27)
        MadNLP.solve!(schur_solver, d)
        MadNLP.solve!(ma27, d_ma27)
        diff = abs.(d - d_ma27)
        maxdiff = maximum(diff)
        println("i = $i, ||ϵ|| = $maxdiff")
    end
#end
#
#main()
