import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys


def plot_bar(csv_path: str, row: int, output: str = None):
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

    series = df.iloc[row]

    # Keep only numeric columns
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        print(f"Error: Row {row} contains no numeric data to plot.")
        sys.exit(1)

    dropped = len(df.columns) - len(series)
    if dropped:
        print(f"Warning: {dropped} non-numeric column(s) in row {row} were ignored.")

    # Build probability bin edges and centre points for bar positions
    n = len(series)
    bin_width = 1.0 / n
    left_edges = [i * bin_width for i in range(n)]
    centres    = [e + bin_width / 2 for e in left_edges]

    # Plot — standard matplotlib figure size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centres, series.values, width=bin_width * 0.9,
           color="steelblue", edgecolor="white", linewidth=0.6)

    ax.set_title(f"Polya Process Step {row+1} - Urn Red Bias Histogram", fontsize=14, pad=12)
    ax.set_xlabel("Red Ball Proportion", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_xlim(0, 1)

    # Fixed ticks at 0.0, 0.1, 0.2, ... 1.0
    ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
    ax.set_xticklabels([f"{i * 0.1:.1f}" for i in range(11)], fontsize=9)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Bar chart saved to '{output}'.")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot all columns of a selected CSV row as a probability histogram."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("row", type=int, help="Zero-based index of the row to plot")
    parser.add_argument(
        "--output",
        default=None,
        help="Save the plot to this file (e.g. chart.png) instead of displaying it",
    )

    args = parser.parse_args()
    plot_bar(args.csv_file, args.row, args.output)


if __name__ == "__main__":
    main()
