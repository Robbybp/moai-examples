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

function make_model()
    m = JuMP.Model()
    JuMP.@variable(m, x[1:3], start = 1.0)
    JuMP.@variable(m, y[1:2], start = 1.0)
    JuMP.set_lower_bound(x[1], 0.0)
    JuMP.set_lower_bound(x[2], 0.0)
    JuMP.set_lower_bound(x[3], -2.0)
    JuMP.set_lower_bound(y[1], 0.0)
    JuMP.set_lower_bound(y[2], 0.0)
    JuMP.set_upper_bound(x[1], 10.0)
    JuMP.set_upper_bound(y[1], 12.0)
    #JuMP.@constraint(m, eq1, x[1] + y[1] + x[2] == 10.0)
    #JuMP.@constraint(m, eq2, 2*x[2] + y[2] - x[3] == 12.0)
    #JuMP.@constraint(m, eq3, y[1] - y[2] == 3.0)
    #JuMP.@constraint(m, eq4, y[2] + 2*y[2] - x[3] == 7.0)
    JuMP.@constraint(m, eq1, x[1]^1.1 + y[1]^1.1 + x[2] == 10.0)
    JuMP.@constraint(m, eq2, 2*x[2] + y[2] - x[3] == 12.0)
    JuMP.@constraint(m, eq3, y[1]^1.1 - y[2] == 3.0)
    JuMP.@constraint(m, eq4, y[2]^1.1 + 2*y[2]^1.1 - x[3] == 7.0)
    JuMP.@constraint(m, ineq1, sum(x) + sum(y) <= 20.0)
    JuMP.@objective(m, Min, sum(x.^2) + sum(y.^2))
    return m
end

function update_kkt!(kkt::MadNLP.AbstractKKTSystem, nlp::NLPModels.AbstractNLPModel)
    # Need to update:
    # - Hessian
    # - Jacobian
    # - Regularization (set to zero? Or leave as default?)
    # - Σ_x, Σ_s (each for upper and lower bounds)
    # For now, I'd like to do the minimum necessary to give me a nonsingular KKT matrix
    hess_values = MadNLP.get_hessian(kkt)
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    #x = NLPModels.get_x0(nlp)
    x = ones(n)
    λ = ones(m)

    NLPModels.hess_coord!(nlp, x, λ, hess_values)

    jac_values = MadNLP.get_jacobian(kkt)
    NLPModels.jac_coord!(nlp, x, jac_values)

    #kkt.reg = 0.0
    #kkt.pr_diag = 0.0
    #kkt.du_diag = 0.0
    return
end

#function main()
    m = make_model()

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
#end
#
#main()
