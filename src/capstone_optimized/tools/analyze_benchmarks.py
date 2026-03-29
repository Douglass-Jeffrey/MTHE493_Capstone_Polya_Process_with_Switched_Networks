import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(file_path, key_col, time_col, mem_col):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(float(row[key_col]))
            time = float(row[time_col])
            mem = float(row[mem_col]) if row[mem_col] else 0.0
            data[key] = {'time': time, 'memory': mem}
    return data

# File paths
cpu_file = 'benchmark_results_CPU_BIG.csv'
gpu_file = 'benchmark_results_GPU_BIG.csv'

# Check if files exist
if not all(os.path.exists(f) for f in [cpu_file, gpu_file]):
    print("One or more CSV files not found.")
    exit(1)

# Load data
cpu_data = load_csv(cpu_file, 'final_nodes', 'total_time', 'cpu_used_mb')
gpu_data = load_csv(gpu_file, 'final_nodes', 'total_time', 'gpu_used_mb')

# Get sorted nodes
nodes = sorted(set(cpu_data.keys()) | set(gpu_data.keys()))

# Prepare data for plotting
cpu_times = [cpu_data.get(n, {'time': 0})['time'] for n in nodes]
cpu_mems = [cpu_data.get(n, {'memory': 0})['memory'] for n in nodes]
gpu_times = [gpu_data.get(n, {'time': 0})['time'] for n in nodes]
gpu_mems = [gpu_data.get(n, {'memory': 0})['memory'] for n in nodes]
prim_times = []
prim_mems = []

# Plot Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(nodes, cpu_times, label='CPU Implementation', marker='o')
plt.plot(nodes, gpu_times, label='GPU Implementation', marker='s')

# set ticks to powers of 2
min_n = min(nodes)
max_n = max(nodes)
pow2_ticks = [2**i for i in range(int(np.floor(np.log2(min_n))), int(np.ceil(np.log2(max_n)))+1)]
plt.xticks(pow2_ticks, labels=[str(t) for t in pow2_ticks])
plt.xscale('log', base=10)
plt.xlabel('Number of Nodes (log10 scale)')
plt.ylabel('Total Time (s)')
plt.yscale('log')
plt.title('Compute Time Comparison')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('time_comparison.png')
plt.show()

# Plot Memory Usage Comparison
plt.figure(figsize=(10, 6))
plt.plot(nodes, cpu_mems, label='CPU Implementation (CPU Memory)', marker='o')
plt.plot(nodes, gpu_mems, label='GPU Implementation (GPU Memory)', marker='s')
plt.xscale('log')
plt.xticks(pow2_ticks, labels=[str(t) for t in pow2_ticks])
plt.xlabel('Number of Nodes (log2 scale)')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('memory_comparison.png')
plt.show()

# Print summary
print("Time and Memory Comparison:")
print(f"{'Nodes':<6} {'CPU Time':<10} {'GPU Time':<10} {'CPU Mem':<10} {'GPU Mem':<10}")
for n in nodes:
    ct = cpu_data.get(n, {'time': 0})['time']
    gt = gpu_data.get(n, {'time': 0})['time']
    cm = cpu_data.get(n, {'memory': 0})['memory']
    gm = gpu_data.get(n, {'memory': 0})['memory']
    print(f"{n:<6} {ct:<10.2f} {gt:<10.2f} {cm:<10.2f} {gm:<10.2f}")

# Speedups
print("\nSpeedups (higher is better):")
print(f"{'Nodes':<6} {'GPU vs CPU':<12}")
for n in nodes:
    ct = cpu_data.get(n, {'time': 0})['time']
    gt = gpu_data.get(n, {'time': 0})['time']
    gpu_cpu = ct / gt if gt > 0 else 0
    print(f"{n:<6} {gpu_cpu:<12.2f}")

print("\nPlots saved as time_comparison.png and memory_comparison.png")