import csv

def compare_columns(filename):
    """
    Compare column 1 and column 2 in a CSV file.
    Returns the proportion of rows where column 1 > column 2.
    """
    col1_greater = 0
    total_rows = 0
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    col1 = float(row[0])
                    col2 = float(row[1])
                    total_rows += 1
                    
                    if col1 > col2:
                        col1_greater += 1
                except ValueError:
                    # Skip rows that can't be converted to float
                    continue
    
    if total_rows == 0:
        print("No valid rows found")
        return None
    
    proportion = col1_greater / total_rows
    
    print(f"Total rows: {total_rows}")
    print(f"Rows where column 1 > column 2: {col1_greater}")
    print(f"Proportion: {proportion:.4f} ({proportion * 100:.2f}%)")
    
    return proportion

if __name__ == "__main__":
    compare_columns("node_urns.csv")
