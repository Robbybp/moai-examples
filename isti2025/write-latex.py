import pandas as pd
import argparse
import os
# config.py should be a file in the working directory (`moai-examples`, not the
# `isti2025` subdirectory.
import importlib.util
import sys
config_fpath = os.path.join(os.path.dirname(__file__), os.pardir, "config.py")
spec = importlib.util.spec_from_file_location("config", config_fpath)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)


COL_HEADER_MAP = {
    "fname": "Model",
    "n_inputs": "N. inputs",
    "n_outputs": "N. outputs",
    "n_neurons": "N. neurons",
    "n_parameters": "N. param.",
    "n_layers": "N. layers",
    "activations": "Activation".rjust(15),
    "NN": "NN".rjust(9),
    "model": "Model",
    "formulation": "Formulation".rjust(13),
    "nvar": "N. var.",
    "ncon": "N. con.",
    "jnnz": "Jac. NNZ",
    "hnnz": "Hess. NNZ",
    "device": "Platform",
    "t_solve_total": "Solve time (s)",
    "n_iterations": "N. iter",
    "time_per_iteration": "Time/iter. (s)",
    "objective_value": "Objective",
    "percent-function": "Function",
    "percent-jacobian": "Jacobian",
    "percent-hessian": "Hessian",
    "percent-solver": "Solver",

    "solver": "Solver",
    "nn": "NN param.".rjust(9),
    "nn-param": "NN param.".rjust(9),
    "t_init": "Initialize",
    "t_factorize": "Factorize",
    "t_solve": "Backsolve",
    "speedup": "Speedup",
    "sample": "Sample",
    "residual": "Residual",
    "refine_iter": "Refinement iter.",

    "matrix_type": "Matrix",
    "dim": "Matrix dim.",
    "nrow": "N. row",
    "ncol": "N. col",
    "nnz": " NNZ",
    "factor_size": "Factor NNZ",
    "flops": "FLOPS",
    "n2by2": "N. 2$\\times$2 pivots",

    "%-construct-schur": "Build Schur",
    "%-factorize-schur": "Factorize Schur",
    "%-factorize-pivot": "Factorize pivot",
    "%-resid": "Compute residual",
    "%-solve-schur": "Solve Schur",
    "%-solve-pivot": "Solve pivot",
}
def _col_to_header(col):
    if col in COL_HEADER_MAP:
        return COL_HEADER_MAP[col]
    else:
        return col


def _fname_to_model(fname):
    if fname.startswith("mnist"):
        return "MNIST"
    elif fname.startswith("scopf"):
        return "SCOPF"
    elif fname.startswith("lsv"):
        return "  LSV"


def _format_int(n):
    n = round(n)
    if n >= 10**9:
        return str(n // 10**9)+"B"
    elif n >= 10**6:
        return str(n // 10**6)+"M"
    elif n >= 10**3:
        return str(n // 10**3)+"k"
    else:
        return str(n)


def _parse_activations(act_str):
    activations = []
    possible_activations = ["Tanh", "Sigmoid", "SoftMax", "SoftPlus", "ReLU"]
    for act in possible_activations:
        if act in act_str:
            activations.append(act)
    return "+".join(activations)


def _parse_solver(name):
    if "Ma57" in name:
        return "MA57"
    elif "Ma86" in name:
        return "MA86"
    elif "Ma97" in name:
        return "MA97"
    elif "Ma27" in name:
        return "MA27"
    elif name == "SchurComplementSolver":
        return "Ours"
    else:
        raise ValueError("Unknown solver")


# TODO: Ideally, this would compute the number of parameters from the NN file
# (or look it up in some other file) rather than hard-coding it here.
def _get_nparam(fname):
    nndir = config.get_nn_dir()
    nnfile = os.path.join(nndir, fname)
    import torch
    nn = torch.load(nnfile, weights_only = False)
    params = [p for p in nn.parameters() if p.requires_grad]
    return sum(p.numel() for p in params)
    #if "mnist" in fname:
    #    if "128nodes" in fname:
    #        return 167818
    #    if "512nodes" in fname:
    #        return 1457674
    #    if "1024nodes" in fname:
    #        return 5012490
    #    if "2048nodes" in fname:
    #        return 18413578
    #    if "4096nodes" in fname:
    #        return 70381578
    #    if "8192nodes" in fname:
    #        return 274980874
    #else:
    #    if "100nodes3layers" in fname:
    #        return 15537
    #    if "500nodes5layers" in fname:
    #        return 578537
    #    if "1000nodes7layers" in fname:
    #        return 4159037
    #    if "1500nodes10layers" in fname:
    #        return 15993037
    #    if "2000nodes20layers" in fname:
    #        return 68344037
    #    if "4000nodes40layers" in fname:
    #        return 592768037
    #raise ValueError("Unrecognized NN")


def _parse_formulation(form):
    if form == "full_space":
        return "Full-space"
    if form == "gray_box":
        return form
    if form == "vector_nonlinear_oracle":
        return "Reduced-space"
    raise ValueError("Unrecognized formulation")


def _parse_matrix_type(matrix):
    match matrix:
        case "original":
            return "KKT"
        case "pivot":
            return "Pivot"
        case "schur":
            return "Schur"
        case "A":
            return "$A$"
        case "B":
            return "$B$"
        case _:
            raise ValueError("unknown matrix type")


def _format_time(n):
    # For times, precision for small numbers is not important
    # or meaningful
    if n > 0.9:
        return str(round(n))
    elif n > 0.09:
        return str(round(n, 1))
    elif n > 0.005:
        return str(round(n, 2))
    else:
        return "$< 0.01$"


def _format_float(n):
    if n is None:
        return "--"
    # We want higher precision for small-ish numbers, here
    elif n >= 10:
        return str(round(n))
    elif n >= 1:
        return str(round(n, 1))
    elif n >= 0.1:
        return str(round(n, 2))
    else:
        return "%1.1E" % n


def _sort_col(c):
    match c.name:
        case "model":
            order = {"mnist": 0, "scopf": 1, "lsv": 2}
            return c.map(order)
        case "solver":
            # Just sort solvers alphabetically
            return c
        case "nn":
            return c.map(_get_nparam)
        case "matrix_type":
            order = {"original": 0, "A": 1, "B": 2, "pivot": 3, "schur": 4}
            return c.map(order)
        case _:
            raise ValueError()


COL_FORMATTER = {
    "model": lambda s: s.upper().rjust(len(COL_HEADER_MAP["model"])),
    "fname": _fname_to_model,
    "n_inputs": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_inputs"])),
    "n_outputs": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_outputs"])),
    "n_neurons": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_neurons"])),
    "n_parameters": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_parameters"])),
    "n_layers": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_layers"])),
    "activations": lambda a: _parse_activations(a).rjust(15),
    "NN": lambda n: _format_int(_get_nparam(n)).rjust(9),
    "nn": lambda n: _format_int(_get_nparam(n)).rjust(9),
    "formulation": lambda n: _parse_formulation(n).rjust(13),
    "nvar": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["nvar"])),
    "ncon": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["ncon"])),
    "jnnz": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["jnnz"])),
    "hnnz": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["hnnz"])),
    "device": lambda dev: ("CPU" if dev=="cpu" else "CPU+GPU").rjust(8),
    "t_solve_total": lambda n: _format_time(n).rjust(14),
    "n_iterations": lambda n: str(n).rjust(7),
    "objective_value": lambda v: _format_float(v).rjust(9),

    "solver": lambda n: _parse_solver(n).rjust(6),
    "t_init":      lambda n: _format_time(n).rjust(10),
    "t_factorize": lambda n: _format_time(n).rjust(9),
    "t_solve":     lambda n: _format_time(n).rjust(9),
    "sample": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["sample"])),
    "residual": lambda n: _format_float(n).rjust(8),
    "refine_iter": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["refine_iter"])),
    "nn-param": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["nn-param"])),

    "matrix_type": lambda n: _parse_matrix_type(n).rjust(len(COL_HEADER_MAP["matrix_type"])),
    "dim": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["dim"])),
    "nrow": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["nrow"])),
    "ncol": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["ncol"])),
    "nnz": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["nnz"])),
    "factor_size": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["factor_size"])),
    "flops": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["flops"])),
    "n2by2": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n2by2"])),
}
def _format_item(col, item):
    return COL_FORMATTER.get(col, str)(item)


CALCULATE_FROM_ROW = {
    "time_per_iteration": lambda row: _format_time(row["t_solve_total"] / row["n_iterations"] if row["n_iterations"] > 0 else float("nan")).rjust(14),
    "percent-function": lambda row: _format_time(row["t_eval_function"] / row["t_solve_total"]*100).rjust(8),
    "percent-jacobian": lambda row: _format_time(row["t_eval_jacobian"] / row["t_solve_total"]*100).rjust(8),
    "percent-hessian": lambda row: _format_time(row["t_eval_hessian"] / row["t_solve_total"]*100).rjust(7),
    "percent-solver": lambda row: _format_time((row["t_solve_total"] - row["t_eval_function"] - row["t_eval_jacobian"] - row["t_eval_hessian"]) / row["t_solve_total"]*100).rjust(6),
    # TODO: format these percents
    "%-construct-schur": lambda row: _format_int(row["construct_schur"] / row["t_factorize"] * 100).rjust(len(COL_HEADER_MAP["%-construct-schur"])),
    "%-factorize-schur": lambda row: _format_int(row["factorize_schur"] / row["t_factorize"] * 100).rjust(len(COL_HEADER_MAP["%-factorize-schur"])),
    "%-factorize-pivot": lambda row: _format_int(row["factorize_pivot"] / row["t_factorize"] * 100).rjust(len(COL_HEADER_MAP["%-factorize-pivot"])),
    "%-resid": lambda row: _format_int(row["compute_resid"] / row["t_solve"] * 100).rjust(len(COL_HEADER_MAP["%-resid"])),
    "%-solve-schur": lambda row: _format_int(row["solve_schur"] / row["t_solve"] * 100).rjust(len(COL_HEADER_MAP["%-solve-schur"])),
    "%-solve-pivot": lambda row: _format_int(row["solve_pivot"] / row["t_solve"] * 100).rjust(len(COL_HEADER_MAP["%-solve-pivot"])),
}


def _calculate_speedup(df, row):
    # Speedup of this row's factorization + backsolve compared to MA57
    # Note that parsing the entire DF here leads to O(n^2) time
    if row["solver"] in ("MadNLPHSL.Ma57Solver", "MadNLPHSL.Ma86Solver"):
        return None
    t_row = row["t_factorize"] + row["t_solve"]
    solver_rows = (df["solver"] == "MadNLPHSL.Ma57Solver") | (df["solver"] == "MadNLPHSL.Ma86Solver")
    try:
        baseline_row = df[df["model"] == row["model"]][df["nn"] == row["nn"]][df["sample"]==row["sample"]][solver_rows].iloc[0]
    except KeyError:
        # ...
        baseline_row = df[df["model"] == row["model"]][df["nn-param"] == row["nn-param"]][solver_rows].iloc[0]
    t_baseline = baseline_row["t_factorize"] + baseline_row["t_solve"]
    speedup = t_baseline / t_row
    return speedup


CALCULATE_FROM_DF = {
    "speedup": lambda df, row: _format_float(_calculate_speedup(df, row)).rjust(7),
}


def df_to_latex(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    lines = [[r"\toprule"]]
    lines.append(list(map(_col_to_header, columns)))
    lines.append([r"\midrule"])
    for i, row in df.iterrows():
        line = []
        for col in columns:
            if col in row:
                item_str = _format_item(col, row[col])
            elif col in CALCULATE_FROM_DF:
                item_str = CALCULATE_FROM_DF[col](df, row)
            else:
                # Calculate the value from this row of the table
                item_str = CALCULATE_FROM_ROW[col](row)
            line.append(item_str)
        lines.append(line)
    lines.append([r"\bottomrule"])

    lines_str = [" & ".join(line) + " \\\\\n" if not line[-1].endswith("rule") else "".join(line)+"\n" for line in lines]
    table_str = "".join(lines_str)
    return table_str


def _nns_df_to_latex(df):
    columns = ["fname", "n_inputs", "n_outputs", "n_neurons", "n_layers", "n_parameters", "activations"]
    return df_to_latex(df, columns=columns)


def _structure_df_to_latex(df):
    df = df.sort_values(by=["model"], key=_sort_col)
    columns = ["model", "n_parameters", "nvar", "ncon", "jnnz", "hnnz"]
    return df_to_latex(df, columns=columns)


def _runtime_df_to_latex(df):
    df = df.sort_values(by=["model", "solver"], key=_sort_col)
    columns = ["model", "solver", "nn", "sample", "t_init", "t_factorize", "t_solve", "residual", "refine_iter", "speedup"]
    return df_to_latex(df, columns=columns)


def _runtime_summary_df_to_latex(df):
    df = df.sort_values(by=["model", "solver"], key=_sort_col)
    columns = ["model", "solver", "nn-param", "t_init", "t_factorize", "t_solve", "residual", "refine_iter", "speedup"]
    return df_to_latex(df, columns=columns)


def _fillin_to_latex(df):
    columns = ["model", "solver", "nn", "matrix_type", "dim", "nnz", "factor_size", "flops", "n2by2"]
    return df_to_latex(df, columns=columns)


def _solvers_to_latex(df):
    df = df.sort_values(by=["model", "solver"], key=_sort_col)
    columns = ["model", "solver", "nn", "t_init", "t_factorize", "t_solve", "residual"]
    return df_to_latex(df, columns=columns)


def _matrix_structure_to_latex(df):
    df = df.sort_values(by=["model", "nn", "matrix_type"], key=_sort_col)
    columns = ["model", "nn", "matrix_type", "nrow", "ncol", "nnz"]
    return df_to_latex(df, columns=columns)


def _breakdown_to_latex(df):
    df = df.sort_values(by="model", key=_sort_col)
    columns = [
        "model",
        "nn-param",
        "t_factorize",
        "%-construct-schur",
        "%-factorize-schur",
        "%-factorize-pivot",
        "t_solve",
        "%-resid",
        "%-solve-schur",
        "%-solve-pivot",
    ]
    return df_to_latex(df, columns=columns)


def main(args):
    if not args.input_fpath.endswith(".csv") and not args.input_fpath.endswith(".CSV"):
        raise ValueError("Input fpath must end with '.csv' or '.CSV'")
    def _write_table(args, table, suffix=""):
        # Replace .csv extension with .txt
        output_fpath = args.input_fpath[:-4] + f"{suffix}.txt"
        if args.dry_run:
            print(f"--dry-run set. Would have written to {output_fpath}")
        else:
            print(f"Writing table to {output_fpath}")
            with open(output_fpath, "w") as f:
                f.write(table)
        print(table)
    df = pd.read_csv(args.input_fpath)
    keys = ["nns", "structure", "runtime-summary", "runtime", "fill-in", "breakdown-summary", "linear-solvers", "matrix-structure"]
    def _name_contains(key, fpath):
        return (
            key in fpath
            and all((k not in fpath) for k in keys if k != key and k not in key)
        )
    if _name_contains("nns", args.input_fpath):
        table_str = _nns_df_to_latex(df)
    elif _name_contains("structure", args.input_fpath):
        table_str = _structure_df_to_latex(df)
    elif _name_contains("runtime-summary", args.input_fpath):
        table_str = _runtime_summary_df_to_latex(df)
    elif _name_contains("runtime", args.input_fpath):
        table_str = _runtime_df_to_latex(df)
    elif _name_contains("fill-in", args.input_fpath):
        table_str = _fillin_to_latex(df)
    elif _name_contains("breakdown-summary", args.input_fpath):
        table_str = _breakdown_to_latex(df)
    elif _name_contains("linear-solvers", args.input_fpath):
        table_str = _solvers_to_latex(df)
    elif _name_contains("matrix-structure", args.input_fpath):
        table_str = _matrix_structure_to_latex(df)
    else:
        raise ValueError("Cannot infer type of table from filename")
    _write_table(args, table_str)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_fpath", help="Input CSV file")
    argparser.add_argument("--dry-run", action="store_true", help="Don't save anything")
    args = argparser.parse_args()
    main(args)
