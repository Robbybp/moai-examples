import pandas as pd
import argparse


COL_HEADER_MAP = {
    "fname": "Model",
    "n_inputs": "N. inputs",
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


COL_FORMATTER = {
    "fname": _fname_to_model,
}
def _format_item(col, item):
    return COL_FORMATTER.get(col, str)(item)


def df_to_latex(df, columns=None):
    if columns is None:
        columns = list(df.columns)
    lines = [list(map(_col_to_header, columns))]
    for i, row in df.iterrows():
        line = []
        for col in columns:
            item_str = _format_item(col, row[col])
            line.append(item_str)
        lines.append(line)

    lines_str = [" & ".join(line) + "\n" for line in lines]
    table_str = "".join(lines_str)
    return table_str


def main(args):
    if not args.input_fpath.endswith(".csv") and not args.input_fpath.endswith(".CSV"):
        raise ValueError("Input fpath must end with '.csv' or '.CSV'")
    df = pd.read_csv(args.input_fpath)
    table_str = df_to_latex(df)
    print(table_str)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_fpath", help="Input CSV file")
    args = argparser.parse_args()
    main(args)
