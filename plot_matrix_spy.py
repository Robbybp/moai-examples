#!/usr/bin/env python3
"""Generate spy plots for the -small matrices in the matrices folder."""
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import mmread

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16

FILEDIR = os.path.dirname(__file__)
MATRIX_DIR = os.path.join(FILEDIR, "matrices")
PROBLEMS = ["mnist", "scopf", "lsv"]
MATRIX_VARIANTS = [
    ("original", "KKT"),
    ("pivot", "Pivot"),
    ("schur", "Schur complement"),
]

def load_matrix(problem: str, variant: str):
    path = os.path.join(MATRIX_DIR, f"{problem}-{variant}-small.mtx")
    matrix = mmread(path)
    return path, matrix


def symmetrize_lower_triangular(matrix: sparse.spmatrix) -> sparse.spmatrix:
    """Return a symmetric matrix assuming the input stores only the lower triangle."""
    matrix = matrix.tocsr()
    diag = sparse.diags(matrix.diagonal())
    return matrix + matrix.T - diag

def plot_problem(problem: str):
    fig, axes = plt.subplots(
        1,
        len(MATRIX_VARIANTS),
        figsize=(12, 4),
        dpi=300,
        constrained_layout=True,
    )

    for ax, (variant, label) in zip(axes, MATRIX_VARIANTS):
        path, matrix = load_matrix(problem, variant)
        symmetric_matrix = symmetrize_lower_triangular(matrix).tocoo()
        line = ax.spy(symmetric_matrix, markersize=0.3)
        line.set_rasterized(True)
        ax.set_title(
            f"{label}\n{symmetric_matrix.shape[0]} x {symmetric_matrix.shape[1]} nnz={symmetric_matrix.nnz}",
            #fontsize=9,
        )
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.grid(False)

    #fig.suptitle(f"{problem.upper()} matrices", fontsize=12)
    outfile = os.path.join(MATRIX_DIR, f"{problem}-small-spy.pdf")
    fig.savefig(outfile, transparent=True)
    plt.close(fig)
    print(f"Wrote {outfile}")

def main():
    for problem in PROBLEMS:
        plot_problem(problem)

if __name__ == "__main__":
    main()
