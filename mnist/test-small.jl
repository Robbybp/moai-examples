import MathOptAI as MOAI
import JuMP
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays

Random.seed!(101)

function block_triangularize(matrix::SparseArrays.SparseMatrixCSC)
    igraph = MPIN.IncidenceGraphInterface(matrix)
    blocks = MPIN.block_triangularize(igraph)
    roworder = reduce(vcat, [r for (r, c) in blocks])
    colorder = reduce(vcat, [c for (r, c) in blocks])
    return roworder, colorder
end

optimizer = JuMP.optimizer_with_attributes(
    MadNLP.Optimizer,
    "tol" => 1e-6,
    "linear_solver" => MadNLPHSL.Ma27Solver,
)

include("models.jl")
m, info = make_small_nn_model()
formulation = info.formulation
include("linalg.jl")

solve = false
if solve
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
end

nlp = NLPModelsJuMP.MathOptNLPModel(m)

include("formulation.jl")
nn_vars, nn_cons = get_vars_cons(formulation)

display_bt_nn_system = false
if display_bt_nn_system
    igraph = MPIN.IncidenceGraphInterface(m)
    blocks = MPIN.block_triangularize(nn_cons, nn_vars)
    cons = vcat([b[1] for b in blocks]...)
    vars = vcat([b[2] for b in blocks]...)
    imat = MPIN.incidence_matrix(cons, vars)
    display(imat)
end

include("nlpmodels.jl")
# These are the var/con orders in the MathOptNLPModel.
varorder, conorder = get_var_con_order(m)
ScalarFunction = Union{
    MOI.ScalarAffineFunction,
    MOI.ScalarQuadraticFunction,
    MOI.ScalarNonlinearFunction,
}
# If we have vector constraints, we have to do something fancier to get the coordinates
@assert all([MOI.get(m, MOI.ConstraintFunction(), c) isa ScalarFunction for c in conorder])
var_idx_map = Dict{JuMP.VariableRef,Int}(var => i for (i, var) in enumerate(varorder))
con_idx_map = Dict{JuMP.ConstraintRef,Int}(con => i for (i, con) in enumerate(conorder))

nn_vindices = [var_idx_map[v] for v in nn_vars]
nn_cindices = [con_idx_map[c] for c in nn_cons]

# Apply offset of nvar+nslack to get constraints' locations in the KKT system
nvar = length(varorder)
ncon = length(conorder)
ind_cons = MadNLP.get_index_constraints(nlp)
nslack = length(ind_cons.ind_ineq)
println("nvar   = $nvar")
println("nslack = $nslack")
println("ncon   = $ncon")

# Get the coordinates of NN vars/cons in the lower-left Jacobian
# Columns (variables)
nn_kktcols = nn_vindices
# Rows (constraints) are offset by the size of the Hessian block
nn_kktrows = nn_cindices .+ (nvar + nslack)

cb = MadNLP.create_callback(MadNLP.SparseCallback, nlp)
#LinearSolver = SchurComplementSolver
# I can't construct SchurComplementSolver with no options as Ma27Solver will fail
# when initialized with an empty matrix.
LinearSolver = MadNLPHSL.Ma27Solver
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    ind_cons,
    LinearSolver,
)
MadNLP.initialize!(kkt)
# I think this happens after initialize?
update_kkt!(kkt, nlp)
MadNLP.build_kkt!(kkt)
kkt_matrix = MadNLP.get_kkt(kkt)
# TODO: Now we need to update the kkt_matrix

# Extract the Jacobian submatrix we want to exploit.
submatrix = kkt_matrix[nn_kktrows, nn_kktcols]
#display(submatrix)
igraph = MPIN.IncidenceGraphInterface(submatrix)
blocks = MPIN.block_triangularize(igraph)
rows = vcat([b[1] for b in blocks]...)
cols = vcat([b[2] for b in blocks]...)
#display(submatrix[rows, cols])

# Check eigenvalues of blocks in the NN's Jacobian
#import LinearAlgebra
#for (i, b) in enumerate(blocks)
#    println("Block $i")
#    rows, cols = b
#    println("row indices = $rows")
#    println("col indices = $cols")
#    sm = submatrix[rows, cols]
#    eig = LinearAlgebra.eigvals(Matrix(sm))
#    # In this case, eigenvalues are real, but have different signs.
#    println("EVs = $(eig)")
#    #svd = LinearAlgebra.svd(Matrix(sm))
#    #println("SVs = $(svd.S)")
#    #SparseArrays.lu(sm)
#end

indices = collect(zip(nn_kktrows, nn_kktcols))
indices = convert(Vector{Tuple{Int32,Int32}}, indices)
schur_dim = length(indices)
kkt_dim = size(kkt_matrix)[1]
rhs = ones(kkt_dim)

# Solve linear system with MA27
ma27 = MadNLPHSL.Ma27Solver(kkt_matrix)
MadNLP.factorize!(ma27)
kkt_inertia_ma27 = MadNLP.inertia(ma27)
println("Inertia of the full KKT matrix according to MA27:")
println("(pos, zero, neg) = $kkt_inertia_ma27")
d_ma27 = copy(rhs)
MadNLP.solve!(ma27, d_ma27)
d_nnvars_ma27 = d_ma27[nn_kktcols]
d_nncons_ma27 = d_ma27[nn_kktrows]

sym_indices = Vector{Int32}(vcat(nn_kktcols, nn_kktrows))
sym_submatrix = kkt_matrix[sym_indices, sym_indices]
# NOTE: With nonzero values not populated, adding these two matrices
# eliminates zero entries, and we can't compute a maximum matching.
#sym_full = sym_submatrix + sym_submatrix'
#roworder, colorder = block_triangularize(sym_full)
# Due to the δg regularization block, this matrix doesn't decompose.
#sym_full_bt = sym_full[roworder, colorder]
sym_index_set = Set(sym_indices)
reduced_indices = [i for i in 1:kkt_dim if !(i in sym_index_set)]

ma27_schur = MadNLPHSL.Ma27Solver(sym_submatrix)
MadNLP.factorize!(ma27_schur)
inertia = MadNLP.inertia(ma27_schur)
println("Inertia of the symmetric submatrix on which we will pivot, according to MA27")
println("(pos, zero, neg) = $inertia")

test_permuted = false
if test_permuted
    permutation_order = vcat(reduced_indices, sym_indices) 
    permuted_kkt_matrix = kkt_matrix[permutation_order, permutation_order]
    ma27_permuted = MadNLPHSL.Ma27Solver(permuted_kkt_matrix)
    MadNLP.factorize!(ma27_permuted)
    inertia = MadNLP.inertia(ma27_permuted)
    println("Inertia of permuted KKT matrix, according to MA27:")
    println("(pos, zero, neg) = $inertia")
end

test_decomposability = true
if test_decomposability
    # Sanity check that the symmetric matrix that we extract still decomposes (if we're
    # okay with losing symmetry)
    hess_submatrix = kkt_matrix[nn_kktcols, nn_kktcols]
    # This is wrong. If I'm going to make symmetric, avoid double-counting diagonal.
    #hess_submatrix = hess_submatrix + hess_submatrix'
    jac_submatrix = kkt_matrix[nn_kktrows, nn_kktcols]
    lefthalf = vcat(hess_submatrix, jac_submatrix)
    # This was for a sanity check without the Hessian block. Should have same
    # number of pos/neg eval. (We should have this with the Hessian block too...)
    #lefthalf = vcat(SparseArrays.spzeros(schur_dim, schur_dim), jac_submatrix)
    #righthalf = vcat(jac_submatrix', SparseArrays.spzeros(schur_dim, schur_dim))
    righthalf = SparseArrays.spzeros(2*schur_dim, schur_dim)
    pivot_noreg = hcat(lefthalf, righthalf)
    # TODO: If I'm going to block triangularize, I need to construct the full matrix
    #roworder, colorder = block_triangularize(sym)
    #sym_bt = sym[roworder, colorder]
    #igraph = MPIN.IncidenceGraphInterface(sym)
    #blocks = MPIN.block_triangularize(igraph)
    # I'll check the inertia of this matrix (without δ) because I can block-triangularize
    # it as a sanity check
    ma27_schur_noreg = MadNLPHSL.Ma27Solver(pivot_noreg)
    MadNLP.factorize!(ma27_schur_noreg)
    inertia = MadNLP.inertia(ma27_schur_noreg)
    # NOTE: MadNLP's documentation of `inertia` is wrong.
    println("Inertia of the linear system with no regularization nonzeros")
    println("(pos, zero, neg) = $inertia")
    # This system has the same number of positive as negative eigenvalues, and no
    # zero eigenvalues.
end

#println("Pivot matrix, extracted directly from KKT:")
#display(sym_submatrix)
#display(Matrix(sym_submatrix))
#println("Pivot matrix, constructed manually from submatrices, without regularization:")
#display(pivot_noreg)
#display(Matrix(pivot_noreg))
#
#println("Sparse matrices equal: $(sym_submatrix == pivot_noreg)")
#println("Dense matrices equal: $(Matrix(sym_submatrix) == Matrix(pivot_noreg))")

#import LinearAlgebra
#for (i, b) in enumerate(blocks)
#    println("Block $i")
#    rows, cols = b
#    println("row indices = $rows")
#    println("col indices = $cols")
#    sm = sym[rows, cols]
#    svd = LinearAlgebra.svd(Matrix(sm))
#    println("SVs = $(svd.S)")
#    #SparseArrays.lu(sm)
#end

opt = SchurComplementOptions(; pivot_indices = sym_indices)
factorize = true
if factorize
    linear_solver = SchurComplementSolver(
        kkt_matrix;
        opt = opt,
    )
    MadNLP.factorize!(linear_solver)
    inertia = MadNLP.inertia(linear_solver)
    println("Inertia according to SchurComplementSolver")
    println("(pos, zero, neg) = $inertia")
    # Intertia is (81, 0, 57). This looks correct based on our numbers of vars/cons.
    d = copy(rhs)
    MadNLP.solve!(linear_solver, d)

    dx_ma27 = d_ma27[reduced_indices]

    # Extract coordinates in the reduced system
    # Note that these are primal-dual indices
    dx = d[reduced_indices]

    dx_diff = dx - dx_ma27
    d_diff = d - d_ma27

    maxdiff = maximum(abs.(d_diff))
    println("Diff at nominal point: $maxdiff")
end

#for i in 1:10
#    global rhs = rand(kkt_dim)
#    x = rand(nvar)
#    update_kkt!(kkt, nlp; x)
#    # I think this is necessary...
#    MadNLP.build_kkt!(kkt)
#    #kkt_matrix = MadNLP.get_kkt_matrix(kkt_system)
#
#    #display(schur_solver.csc)
#    ## Reduced solver's matrix has not been updated
#    #display(schur_solver.reduced_solver.csc)
#    #display(schur_solver.schur_solver.csc)
#
#    # We should be able to factorize and solve with the same linear solvers
#    global d = copy(rhs)
#    global d_ma27 = copy(rhs)
#    # Use linear_solver, which we created in the above if block
#    MadNLP.factorize!(linear_solver)
#    MadNLP.factorize!(ma27)
#    global schur_inertia = MadNLP.inertia(linear_solver)
#    global ma27_inertia = MadNLP.inertia(ma27)
#    MadNLP.solve!(linear_solver, d)
#    MadNLP.solve!(ma27, d_ma27)
#    diff = abs.(d - d_ma27)
#    maxdiff = maximum(diff)
#    println("i = $i, Schur inertia = $schur_inertia, MA27 inertia = $ma27_inertia, ||ϵ|| = $maxdiff")
#end
