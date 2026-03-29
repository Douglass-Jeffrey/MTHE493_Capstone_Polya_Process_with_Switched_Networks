import argparse
import numpy as np
 
 
def merge_columns(input_path: str, output_path: str, k: int, header = False) -> None:
    data = np.loadtxt(input_path, delimiter=",", skiprows=1 if header else 0)
 
    # Handle 1D edge case (single row)
    if data.ndim == 1:
        data = data.reshape(1, -1)
 
    n_cols = data.shape[1]
    if n_cols % k != 0:
        raise ValueError(
            f"Number of columns ({n_cols}) is not divisible by k={k}."
        )
 
    # Reshape so each group of k columns becomes a depth slice, then sum
    merged = data.reshape(data.shape[0], n_cols // k, k).sum(axis=2)
 
    np.savetxt(output_path, merged, delimiter=",", fmt="%d")
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CSV columns in groups of k by summing.")
    parser.add_argument("input",  help="Path to input CSV file.")
    parser.add_argument("output", help="Path to output CSV file.")
    parser.add_argument("--k",   type=int, required=True, help="Group size to sum over.")
    parser.add_argument("--no-header", dest="header", action="store_false",
                        help="Pass this flag if the CSV has no header row (default: has header).")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    merge_columns(args.input, args.output, args.k, args.header)
 