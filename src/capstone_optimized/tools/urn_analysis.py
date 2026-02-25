import csv
import sys

def urn_analysis(filename, proportion_threshold=0.5):
    col1_greater = 0
    total_rows = 0
    filename = str(filename)
    proportion_threshold = float(proportion_threshold)
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    col1 = float(row[0])
                    col2 = float(row[1])
                    total_rows += 1
                    
                    if col1 / (col1 + col2) > proportion_threshold:
                        col1_greater += 1
                except ValueError:
                    # Skip rows that can't be converted to float
                    continue
    
    if total_rows == 0:
        print("No valid rows found")
        return None
    
    proportion = col1_greater / total_rows
    
    print(f"Total rows: {total_rows}")
    print(f"Rows where Red balls > Black balls(with threshold {proportion_threshold}): {col1_greater}")
    print(f"Proportion: {proportion:.10f} ({proportion * 100:.10f}%)")
    
    return proportion

if __name__ == "__main__":
    urn_analysis(sys.argv[1], sys.argv[2])
