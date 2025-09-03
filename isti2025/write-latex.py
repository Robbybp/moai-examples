import pandas as pd
import argparse


COL_HEADER_MAP = {
    "fname": "Model",
    "n_inputs": "N. inputs",
    "n_outputs": "N. outputs",
    "n_neurons": "N. neurons",
    "n_parameters": "N. param.",
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


def _format_int(n):
    n = round(n)
    if n >= 10**6:
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
    if name == "SchurComplementSolver":
        return "Ours"


def _get_nparam(fname):
    if "mnist" in fname:
        if "128nodes" in fname:
            return 167818
        if "512nodes" in fname:
            return 1457674
        if "1024nodes" in fname:
            return 5012490
        if "2048nodes" in fname:
            return 18413578
        if "4096nodes" in fname:
            return 70381578
        if "8192nodes" in fname:
            return 274980874
    else:
        if "100nodes3layers" in fname:
            return 15537
        if "500nodes5layers" in fname:
            return 578537
        if "1000nodes7layers" in fname:
            return 4159037
        if "1500nodes10layers" in fname:
            return 15993037
        if "2000nodes20layers" in fname:
            return 68344037
        if "4000nodes40layers" in fname:
            return 592768037
    raise ValueError("Unrecognized NN")


def _parse_formulation(form):
    if form == "full_space":
        return "Full-space"
    if form == "gray_box":
        return form
    if form == "vector_nonlinear_oracle":
        return "Reduced-space"
    raise ValueError("Unrecognized formulation")


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


COL_FORMATTER = {
    "model": lambda s: s.upper(),
    "fname": _fname_to_model,
    "n_inputs": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_inputs"])),
    "n_outputs": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_outputs"])),
    "n_neurons": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_neurons"])),
    "n_parameters": lambda n: _format_int(n).rjust(len(COL_HEADER_MAP["n_parameters"])),
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
}
def _format_item(col, item):
    return COL_FORMATTER.get(col, str)(item)


CALCULATE_FROM_ROW = {
    "time_per_iteration": lambda row: _format_time(row["t_solve_total"] / row["n_iterations"] if row["n_iterations"] > 0 else float("nan")).rjust(14),
    "percent-function": lambda row: _format_time(row["t_eval_function"] / row["t_solve_total"]*100).rjust(8),
    "percent-jacobian": lambda row: _format_time(row["t_eval_jacobian"] / row["t_solve_total"]*100).rjust(8),
    "percent-hessian": lambda row: _format_time(row["t_eval_hessian"] / row["t_solve_total"]*100).rjust(7),
    "percent-solver": lambda row: _format_time((row["t_solve_total"] - row["t_eval_function"] - row["t_eval_jacobian"] - row["t_eval_hessian"]) / row["t_solve_total"]*100).rjust(6),
}


def _calculate_speedup(df, row):
    # Speedup of this row's factorization + backsolve compared to MA57
    # Note that parsing the entire DF here leads to O(n^2) time
    if row["solver"] == "MadNLPHSL.Ma57Solver":
        return None
    t_row = row["t_factorize"] + row["t_solve"]
    try:
        baseline_row = df[df["model"] == row["model"]][df["nn"] == row["nn"]][df["sample"]==row["sample"]][df["solver"] == "MadNLPHSL.Ma57Solver"].iloc[0]
    except KeyError:
        # ...
        baseline_row = df[df["model"] == row["model"]][df["nn-param"] == row["nn-param"]][df["solver"] == "MadNLPHSL.Ma57Solver"].iloc[0]
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

    lines_str = [" & ".join(line) + " \\\\\n" for line in lines]
    table_str = "".join(lines_str)
    return table_str


def _nns_df_to_latex(df):
    columns = ["fname", "n_inputs", "n_outputs", "n_neurons", "n_parameters", "activations"]
    return df_to_latex(df, columns=columns)


def _structure_df_to_latex(df):
    df = df.sort_values(by=["model", "formulation"])
    columns = ["model", "formulation", "NN", "nvar", "ncon", "jnnz", "hnnz"]
    return df_to_latex(df, columns=columns)


def _runtime_df_to_latex(df):
    df = df.sort_values(by=["model", "solver"])
    columns = ["model", "solver", "nn", "sample", "t_init", "t_factorize", "t_solve", "residual", "refine_iter", "speedup"]
    return df_to_latex(df, columns=columns)


def _runtime_summary_df_to_latex(df):
    df = df.sort_values(by=["model", "solver"])
    columns = ["model", "solver", "nn-param", "t_init", "t_factorize", "t_solve", "residual", "refine_iter", "speedup"]
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
    if "nns" in args.input_fpath and "structure" not in args.input_fpath and "runtime" not in args.input_fpath:
        table_str = _nns_df_to_latex(df)
    elif "structure" in args.input_fpath and "nns" not in args.input_fpath and "runtime" not in args.input_fpath:
        table_str = _structure_df_to_latex(df)
    elif "runtime-summary" in args.input_fpath and "structure" not in args.input_fpath and "nns" not in args.input_fpath:
        table_str = _runtime_summary_df_to_latex(df)
    elif "runtime" in args.input_fpath and "structure" not in args.input_fpath and "nns" not in args.input_fpath:
        table_str = _runtime_df_to_latex(df)
    else:
        raise ValueError("Cannot infer type of table from filename")
    _write_table(args, table_str)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_fpath", help="Input CSV file")
    argparser.add_argument("--dry-run", action="store_true", help="Don't save anything")
    args = argparser.parse_args()
    main(args)
