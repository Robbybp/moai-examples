import os
import pandas as pd
import argparse

import importlib.util
import sys
config_fpath = os.path.join(os.path.dirname(__file__), os.pardir, "config.py")
spec = importlib.util.spec_from_file_location("config", config_fpath)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)


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


def process_data(df):
    df["nn"] = [_get_nparam(nn) for nn in df["nn"]]
    colset = set(df.columns)
    df = df.groupby(["model", "solver", "nn"])

    combine_methods = {
        't_init': 'first',
        't_factorize': 'sum',
        't_solve': 'sum',
        'nneg_eig': 'mean',
        'residual': 'mean',
        'refine_success': 'count',
        'refine_iter': 'mean',
        # Solvetime breakdowns
        'factorize_schur': 'sum',
        'factorize_pivot': 'sum',
        'construct_schur': 'sum',
        'other_factorize': 'sum',
        'compute_resid': 'sum',
        'solve_schur': 'sum',
        'solve_pivot': 'sum',
        'compute_rhs': 'sum',
        'other_backsolve': 'sum',
    }
    combine_methods = {
        col: combine_methods[col]
        for col in combine_methods if col in colset
    }
    combined_df = df.agg(combine_methods).reset_index()
    combined_df = combined_df.rename(columns={"nn": "nn-param"})
    return combined_df


def main(args):
    if not args.input_fpath.endswith(".csv") and not args.input_fpath.endswith(".CSV"):
        raise ValueError("Input fpath must end with '.csv' or '.CSV'")
    input_df = pd.read_csv(args.input_fpath)
    output_df = process_data(input_df)
    output_fpath = args.input_fpath[:-4] + "-summary" + args.input_fpath[-4:]
    print(output_df)
    if not args.dry_run:
        print(f"Writing output to {output_fpath}")
        output_df.to_csv(output_fpath)
    else:
        print(f"--dry-run set. Would have written to {output_fpath}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_fpath", help="Input CSV file")
    argparser.add_argument("--dry-run", action="store_true", help="Don't save anything")
    args = argparser.parse_args()
    main(args)
