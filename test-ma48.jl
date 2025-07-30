import NLPModels
import SparseArrays
import LinearAlgebra

include("ma48.jl")
include("models.jl")
include("nlpmodels.jl")
include("linalg.jl") # We just need fill_upper_triangle from this file

m, info = make_tiny_model()
_, _, matrix = get_kkt(m)
csc = fill_upper_triangle(matrix)

ma48 = Ma48Solver(csc)
println("INFO after symbolic:")
display(ma48.INFO)
println("RINFO after symbolic:")
display(ma48.RINFO)

MadNLP.factorize!(ma48)
println("INFO after numeric:")
display(ma48.INFO)
println("RINFO after numeric:")
display(ma48.RINFO)

rhs = ones(csc.m, 2)
sol = copy(rhs)
MadNLP.solve!(ma48, sol)
println("Solution from MA48")
display(sol)

lu = LinearAlgebra.lu(csc)
sol_umfpack = lu \ rhs
println("Solution from UMFPACK")
display(sol_umfpack)

@assert all(isapprox.(sol_umfpack, sol, atol=1e-8))
maxdiff = maximum(abs.(sol_umfpack - sol))
println("||ϵ|| = $maxdiff")
