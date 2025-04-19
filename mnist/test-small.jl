import MathOptAI as MOAI
import JuMP
import MadNLP, MadNLPHSL
import MathOptInterface as MOI
import MathProgIncidence as MPIN
import NLPModels, NLPModelsJuMP
import Random
import SparseArrays

Random.seed!(101)
input_dim = 8
hidden_dim = 16
output_dim = 4

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

include("small-model.jl")
m, formulation = make_small_model()
include("linalg.jl")

solve = false
if solve
    JuMP.set_optimizer(m, optimizer)
    JuMP.optimize!(m)
end

nlp = NLPModelsJuMP.MathOptNLPModel(m)

include("formulation.jl")
nn_vars, nn_cons = get_vars_cons(formulation)

igraph = MPIN.IncidenceGraphInterface(m)
blocks = MPIN.block_triangularize(nn_cons, nn_vars)
cons = vcat([b[1] for b in blocks]...)
vars = vcat([b[2] for b in blocks]...)
imat = MPIN.incidence_matrix(cons, vars)
display(imat)

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
kkt_matrix = MadNLP.get_kkt(kkt)
submatrix = kkt_matrix[nn_kktrows, nn_kktcols]
display(submatrix)

igraph = MPIN.IncidenceGraphInterface(submatrix)
blocks = MPIN.block_triangularize(igraph)
rows = vcat([b[1] for b in blocks]...)
cols = vcat([b[2] for b in blocks]...)
display(submatrix[rows, cols])

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
d_ma27 = copy(rhs)
MadNLP.solve!(ma27, d_ma27)
d_nnvars_ma27 = d_ma27[nn_kktcols]
d_nncons_ma27 = d_ma27[nn_kktrows]

sym_indices = vcat(nn_kktcols, nn_kktrows)
sym_submatrix = kkt_matrix[sym_indices, sym_indices]
sym_full = sym_submatrix + sym_submatrix'
roworder, colorder = block_triangularize(sym_full)
# Due to the δg regularization block, this matrix doesn't decompose.
sym_full_bt = sym_full[roworder, colorder]

ma27_schur = MadNLPHSL.Ma27Solver(sym_submatrix)
MadNLP.factorize!(ma27_schur)
inertia = MadNLP.inertia(ma27_schur)
println("(pos, zero, neg) = $inertia")

# Sanity check that the symmetric matrix that we extract still decomposes (if we're
# okay with losing symmetry)
hess_submatrix = kkt_matrix[nn_kktcols, nn_kktcols]
hess_submatrix = hess_submatrix + hess_submatrix'
jac_submatrix = kkt_matrix[nn_kktrows, nn_kktcols]
lefthalf = vcat(hess_submatrix, jac_submatrix)
# This was for a sanity check without the Hessian block. Should have same
# number of pos/neg eval. (We should have this with the Hessian block too...)
#lefthalf = vcat(SparseArrays.spzeros(schur_dim, schur_dim), jac_submatrix)
righthalf = vcat(jac_submatrix', SparseArrays.spzeros(schur_dim, schur_dim))
sym = hcat(lefthalf, righthalf)
#roworder, colorder = block_triangularize(sym)
#sym_bt = sym[roworder, colorder]
#igraph = MPIN.IncidenceGraphInterface(sym)
#blocks = MPIN.block_triangularize(igraph)
# I'll check the inertia of this matrix (without δ) because I can block-triangularize
# it as a sanity check
ma27_schur_noreg = MadNLPHSL.Ma27Solver(sym)
MadNLP.factorize!(ma27_schur_noreg)
inertia = MadNLP.inertia(ma27_schur_noreg)
# NOTE: MadNLP's documentation of `inertia` is wrong.
println("(pos, zero, neg) = $inertia")
# This system has the same number of positive as negative eigenvalues, and no
# zero eigenvalues.

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

factorize = false
if factorize
    opt = SchurComplementOptions(; indices = indices)
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
end
