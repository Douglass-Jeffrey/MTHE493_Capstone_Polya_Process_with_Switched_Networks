import pandas as pd
import argparse
import sys


def print_stats(csv_path: str, row: int):
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)

    # Validate row index
    if row < 0 or row >= len(df):
        print(f"Error: Row index {row} is out of range. File has {len(df)} rows (0–{len(df) - 1}).")
        sys.exit(1)

    series = pd.to_numeric(df.iloc[row], errors="coerce").dropna()
    if series.empty:
        print(f"Error: Row {row} contains no numeric data.")
        sys.exit(1)

    n = len(series)
    bin_width = 1.0 / n
    values = series.values

    # Assign each bin to a decade range (0–0.1, 0.1–0.2, ..., 0.9–1.0)
    decade_sums = {}
    for i in range(10):
        lo = round(i * 0.1, 1)
        hi = round((i + 1) * 0.1, 1)
        decade_values = [
            values[j] for j in range(n)
            if lo <= round(j * bin_width, 10) < hi
        ]
        decade_sums[(lo, hi)] = sum(decade_values)

    total = sum(values)

    # 0.8–1.0 summary
    high_sum = decade_sums[(0.8, 0.9)] + decade_sums[(0.9, 1.0)]
    proportion = high_sum / total if total != 0 else float("nan")

    # LaTeX table
    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \begin{tabular}{cc}")
    print(r"        \hline")
    print(r"        \textbf{Red Ball Proportion} & \textbf{Number of Nodes} \\")
    print(r"        \hline")
    for (lo, hi), s in decade_sums.items():
        print(f"        ${lo:.1f} - {hi:.1f}$ & ${s:.0f}$ \\\\")
    print(r"        \hline")
    print(f"        Number of aware nodes ($0.8 - 1.0$) & ${high_sum:.0f}$ \\\\")
    print(f"        Proportion of aware nodes & ${proportion:.4f}$ \\\\")
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"    \caption{Awareness distribution for Polya process step " + str(row + 1) + r"}")
    print(r"    \label{tab:bin_sums}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Print probability bin range sums for a selected CSV row."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("row", type=int, help="Zero-based index of the row to analyse")

    args = parser.parse_args()
    print_stats(args.csv_file, args.row)


if __name__ == "__main__":
    main()
