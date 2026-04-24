import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python plot_hist_heatmap.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

# Load the CSV where each row is a timestep and each column is a bin
hist_data = pd.read_csv(csv_file, header=None).values

# Normalize by row sums (each timestep sums to 1)
row_sums = hist_data.sum(axis=1, keepdims=True)
hist_norm = hist_data / row_sums

# Compute vmin and vmax as percentiles
vmin_val = np.percentile(hist_norm, 1)
vmax_val = np.percentile(hist_norm, 99)

import os

# Create output filename automatically
output_file = os.path.splitext(csv_file)[0] + "_heatmap.png"

plt.figure(figsize=(12, 6))

plt.imshow(hist_norm.T, aspect='auto', origin='lower', 
            extent=[0, hist_norm.shape[0], 0, 1],
            cmap='magma', vmin=vmin_val, vmax=vmax_val)

plt.colorbar(label='Proportion of Nodes in Network')
plt.title("Heatmap of Red Ball Proportions over Polya Process")
plt.xlabel("Polya Process Steps")
plt.ylabel("Per Urn Red Ball Proportion")
#plt.show()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: {output_file}")
