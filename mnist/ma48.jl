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

    ICNTL::Vector{INT}
    CNTL::Vector{T}

    INFO::Vector{INT}
    RINFO::Vector{T}

    A::Vector{T}
    IRN::Vector{INT}
    JCN::Vector{INT}

    KEEP::Vector{INT}
end

function Ma48Solver(
    # MA48 doesn't have a "generic" wrapper yet, so we only support double and int32
    csc::SparseArrays.SparseMatrixCSC{Float64,Int32};
    min_blocksize::Int32 = Int32(1),
)
    I, J, V = SparseArrays.findnz(csc)

    # Note that I didn't finish implementing these methods for the defaults...
    # I got lazy and decided to just use ma48id...
    #CNTL = ma48_default_cntl(Float64)
    #ICNTL = ma48_default_icntl(Int32)
    ## Make sure I can count...
    #@assert length(CNTL) == 10
    #@assert length(ICNTL) == 20
    CNTL = Vector{Float64}(undef, 10)
    ICNTL = Vector{Int32}(undef, 20)
    HSL.ma48id(CNTL, ICNTL)

    ICNTL[6] = min_blocksize

    JOB = Int32(1) # Compute a pivot sequence; don't restrict pivoting to diagonal
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
    LKEEP = Int32(M + 5*N + floor(4*N/ICNTL[6]) + 7)
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
        A,
        IRN,
        JCN,
        KEEP,
    )
end

function MadNLP.factorize!(solver::Ma48Solver{T,INT}) where {T,INT}
    csc = solver.csc
    _, _, V = SparseArrays.findnz(csc)

    JOB = INT(1) # "normal call", not a "fast call"
    M = INT(csc.m)
    N = INT(csc.n)
    NE = INT(SparseArrays.nnz(csc))
    LA = INT(5 * NE) # Minimum = 2*NE. "Very problem dependent", according to docs
    # TODO: Add a "LA factor" option
    A = Vector{T}(undef, LA)
    IRN = Vector{INT}(undef, LA)
    JCN = Vector{INT}(undef, NE)
    A[1:NE] = V
    # This is what the docs say to do. Don't ask me...
    IRN[1:NE] = solver.IRN[1:NE]
    JCN[1:NE] = solver.JCN[1:NE]
    # TODO: Allocate W and IW once in initialization
    W = Vector{INT}(undef, M)
    LIW = INT(2*M + 2*N)
    IW = Vector{INT}(undef, LIW)
    HSL.ma48bd(
        M,
        N,
        NE,
        JOB,
        LA,
        A,
        IRN,
        JCN,
        solver.KEEP,
        solver.CNTL,
        solver.ICNTL,
        W,
        IW,
        solver.INFO,
        solver.RINFO,
    )
    # TODO: Check for memory error and factorize in a loop
    if solver.INFO[1] != 0
        throw(MadNLP.FactorizationException())
    end
    # TODO: We should just use solver.A directly. This will require allocating
    # the correct amount of memory for A at the end of the constructor above.
    solver.A = A
    solver.IRN = IRN
    # For some reason, JCN isn't necessary to cache for the backsolve.
    return solver
end

function MadNLP.solve!(solver::Ma48Solver{T,INT}, rhs::Vector{T}) where {T,INT}
    csc = solver.csc
    TRANS = INT(0)
    JOB = INT(1) # Whether we want iterative refinement and/or error estimation. (1=>no)
    M = INT(csc.m)
    N = INT(csc.n)
    NE = INT(SparseArrays.nnz(csc))
    # TODO: Most of this work can probably be avoided here. (I.e., arrays can be
    # allocated during initialization.)
    LA = INT(5 * NE)
    X = Vector{T}(undef, N)
    ERROR = Vector{T}(undef, 3)
    LW = INT(3*M + N)
    W = Vector{T}(undef, LW)
    IW = Vector{INT}(undef, M)
    HSL.ma48cd(
        M,
        N,
        TRANS,
        JOB,
        LA,
        solver.A,
        solver.IRN,
        solver.KEEP,
        solver.CNTL,
        solver.ICNTL,
        rhs,
        X,
        ERROR,
        W,
        IW,
        solver.INFO,
    )
    if solver.INFO[1] != 0
        throw(SolveException())
    end
    rhs[1:M] = X # Explicitly overwrite the RHS, since apparently this isn't done automatically
    return X
end

function MadNLP.solve!(solver::Ma48Solver{T,INT}, rhs::Matrix{T}) where {T,INT}
    rhsdim, nrhs = rhs.size
    for j in 1:nrhs
        temp = rhs[:, j]
        MadNLP.solve!(solver, temp)
        rhs[:, j] = temp
    end
    return rhs
end
