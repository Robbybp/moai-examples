import HSL
import MadNLP
import SparseArrays

"""MA48 interface.

Arrays lengths and default values are taken from the MA48 documentation:
https://www.hsl.rl.ac.uk/specs/ma48.pdf
"""

ma48_default_cntl(T) = T[
    0.5, # Fraction of full-matrix entries when we switch from sparse to dense
    0.1, # Pivot threshold. Presumably the minimum pivot value, as a fraction of the max entry.
    0.0, # Zero threshold. Anything lower in magnitude, we ignore.
    0.0, # Null pivot threshold.
    0.5, # Iterative refinement required improvement factor
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
ma48_default_icntl(INT) = INT[
    6, # Where to send error messages. 6 is Fortran for stderr?
    6, # Where to send diagnostic messages.
    2, # Print level. 2 => errors and warnings.
    3, # N. columns to search for pivot
    32, # L3 BLAS control
    1, # Minimum size of block (in BTF) other than final block.
    1, # Whether to handle structurally rank-deficient matrices. 1=>no, 0=>yes.
    0,
]

mutable struct Ma48Solver{T,INT} <: MadNLP.AbstractLinearSolver{T}
    csc::SparseArrays.SparseMatrixCSC{T,INT}

    icntl::Vector{INT}
    cntl::Vector{T}

    info::Vector{INT}
    rinfo::Vector{T}

    keep::Vector{INT}
end

function Ma48Solver(
    # MA48 doesn't have a "generic" wrapper yet, so we only support double and int32
    csc::SparseArrays.SparseMatrixCSC{Float64,Int32},
)
    I, J, V = SparseArrays.findnz(csc)

    # TODO: Populate reasonable cntl/icntl
    #CNTL = ma48_default_cntl(Float64)
    #ICNTL = ma48_default_icntl(Int32)
    ## Make sure I can count...
    #@assert length(CNTL) == 10
    #@assert length(ICNTL) == 20
    CNTL = Vector{Float64}(undef, 10)
    ICNTL = Vector{Int32}(undef, 20)
    HSL.ma48id(CNTL, ICNTL)

    JOB = Int32(1) # Compute a pivot sequence; don't restricti pivoting to diagonal
    M = Int32(csc.m)
    N = Int32(csc.n)
    NE = Int32(SparseArrays.nnz(csc))
    LA = Int32(3 * NE) # Minimum = 2*NE. 3*NE is recommended.
    A = Vector{Float64}(undef, LA)
    IRN = Vector{Int32}(undef, LA)
    JCN = Vector{Int32}(undef, LA)
    A[1:NE] = V
    IRN[1:NE] = I
    JCN[1:NE] = J
    LKEEP = Int32(M + 5*N + 4*floor(N/ICNTL[6]) + 7)
    KEEP = Vector{Int32}(undef, LKEEP)
    LIW = Int32(6*M + 3*N)
    # Otherwise, we must initialize IW, communicating the columns that are
    # unchanged since the previous factorization.
    @assert ICNTL[8] == 0
    IW = Vector{Int32}(undef, LIW)
    INFO = Vector{Int32}(undef, 20)
    RINFO = Vector{Float64}(undef, 10)
    HSL.ma48ad(
        M,
        N,
        NE,
        JOB,
        LA,
        A,
        IRN,
        JCN,
        KEEP,
        CNTL,
        ICNTL,
        IW,
        INFO,
        RINFO,
    )
    if INFO[1] < 0
        throw(MadNLP.SymbolicException())
    end
    return Ma48Solver{Float64,Int32}(
        csc,
        ICNTL,
        CNTL,
        INFO,
        RINFO,
        KEEP,
    )
end

function MadNLP.factorize!(M::Ma48Solver{T,INT}) where {T,INT}
end

function MadNLP.solve!(M::Ma48Solver{T,INT}, rhs::Vector{T}) where {T,INT}
end

function MadNLP.solve!(M::Ma48Solver{T,INT}, rhs::Matrix{T}) where {T,INT}
end
