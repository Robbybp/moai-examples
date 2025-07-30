# This runs some performance/profile tests that might be too time-consuming
# if you just want to test that the code works.
include("test-blockdiagonal.jl")
include("test-btsolver.jl")
include("test-linalg.jl")
include("profile-linalg.jl")
include("test-nlp.jl")
