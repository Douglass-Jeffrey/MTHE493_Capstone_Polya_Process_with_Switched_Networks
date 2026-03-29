"""
analyze_benchmarks.py
---------------------
Compares compute time and peak memory for CPU, local GPU, and A100 benchmarks.

Works with both the OLD CSV format (cpu_used_mb / cpu_peak_mb columns that
may contain negative or carry-forward values) and the NEW format produced by
the fixed test_benchmark.py (cpu_peak_delta_mb which is always >= 0).
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(file_path):
    """Load every row of a CSV as a list of dicts (all values as strings)."""
    with open(file_path, 'r') as f:
        return list(csv.DictReader(f))


def to_float(val, default=None):
    """Convert a string to float; return default on failure or empty string."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def aggregate_by_nodes(rows, time_col, mem_fn):
    """
    Group rows by final_nodes.  For each node count keep:
      - minimum total_time  (best-case across batch configs)
      - maximum memory      (worst-case / peak across batch configs)

    Parameters
    ----------
    rows    : list of dict (raw CSV rows)
    time_col: column name for the time value
    mem_fn  : callable(row) -> float|None — extracts the memory metric

    Returns
    -------
    dict: { node_count (int): {'time': float, 'memory': float} }
    """
    grouped = {}
    for row in rows:
        n = int(float(row['final_nodes']))
        t = to_float(row.get(time_col), default=None)
        m = mem_fn(row)
        if t is None:
            continue
        if n not in grouped:
            grouped[n] = {'time': t, 'memory': m if m is not None else 0.0}
        else:
            grouped[n]['time'] = min(grouped[n]['time'], t)
            if m is not None:
                grouped[n]['memory'] = max(grouped[n]['memory'], m)
    return grouped


# ---------------------------------------------------------------------------
# Memory extraction strategies
# ---------------------------------------------------------------------------

def cpu_memory(row):
    """
    Extract peak CPU memory delta for one row, handling both CSV formats.

    New format  : cpu_peak_delta_mb  (always >= 0, direct delta from baseline)
    Old format  : cpu_mem_growth - cpu_mem_init
                  (growth-phase RSS minus pre-benchmark baseline; clamped to 0
                   to remove any spurious negatives from GC noise)
    """
    # New format
    v = to_float(row.get('cpu_peak_delta_mb'))
    if v is not None:
        return max(0.0, v)

    # Old format — use growth snapshot minus init (avoids end-of-run GC dip)
    growth = to_float(row.get('cpu_mem_growth'))
    init   = to_float(row.get('cpu_mem_init'))
    if growth is not None and init is not None:
        return max(0.0, growth - init)

    # Final fallback: signed delta, clamped
    final = to_float(row.get('cpu_mem_final'))
    if final is not None and init is not None:
        return max(0.0, final - init)

    return None


def gpu_memory(row):
    """
    Extract peak GPU memory delta for one row.

    For GPU/A100 rows the reliable metric is the CuPy memory-pool delta
    (gpu_used_pool_mb), which is always a non-negative delta in both formats.
    """
    v = to_float(row.get('gpu_used_pool_mb'))
    if v is not None:
        return max(0.0, v)
    return None


# ---------------------------------------------------------------------------
# Load files
# ---------------------------------------------------------------------------

CPU_FILE = 'benchmark_results_CPU_BIG_2.csv'
GPU_FILE = 'benchmark_results_GPU_BIG_3.csv'
A100_FILE = 'benchmark_results_A100_BIG_csv.csv'

missing = [f for f in [CPU_FILE, GPU_FILE, A100_FILE] if not os.path.exists(f)]
if missing:
    print(f"Missing file(s): {missing}")
    exit(1)

cpu_rows  = load_csv(CPU_FILE)
gpu_rows  = load_csv(GPU_FILE)
a100_rows = load_csv(A100_FILE)

cpu_data  = aggregate_by_nodes(cpu_rows,  'total_time', cpu_memory)
gpu_data  = aggregate_by_nodes(gpu_rows,  'total_time', gpu_memory)
a100_data = aggregate_by_nodes(a100_rows, 'total_time', gpu_memory)

all_nodes = sorted(
    set(cpu_data.keys()) | set(gpu_data.keys()) | set(a100_data.keys())
)

# Build aligned arrays (0 for missing entries so curves still plot)
def vals(data, key):
    return [data.get(n, {key: 0.0})[key] for n in all_nodes]

cpu_times  = vals(cpu_data,  'time')
gpu_times  = vals(gpu_data,  'time')
a100_times = vals(a100_data, 'time')

cpu_mems   = vals(cpu_data,  'memory')
gpu_mems   = vals(gpu_data,  'memory')
a100_mems  = vals(a100_data, 'memory')


# ---------------------------------------------------------------------------
# Axis tick helpers
# ---------------------------------------------------------------------------

def pow2_ticks(nodes):
    """Return a list of powers-of-2 spanning the node range."""
    mn, mx = min(nodes), max(nodes)
    lo = int(np.floor(np.log2(mn)))
    hi = int(np.ceil(np.log2(mx)))
    return [2 ** i for i in range(lo, hi + 1)]


def apply_log2_xaxis(ax, nodes):
    """Use log2 x-axis with power-of-2 tick labels."""
    ticks = pow2_ticks(nodes)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlim(min(nodes) * 0.8, max(nodes) * 1.3)


MARKERS = {'cpu': 'o', 'gpu': 's', 'a100': '^'}
COLORS  = {'cpu': '#1f77b4', 'gpu': '#ff7f0e', 'a100': '#2ca02c'}


# ---------------------------------------------------------------------------
# Plot 1 — Compute Time
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(all_nodes, cpu_times,  label='CPU',  marker=MARKERS['cpu'],
        color=COLORS['cpu'],  linewidth=1.8)
ax.plot(all_nodes, gpu_times,  label='Local GPU',  marker=MARKERS['gpu'],
        color=COLORS['gpu'],  linewidth=1.8)
ax.plot(all_nodes, a100_times, label='A100', marker=MARKERS['a100'],
        color=COLORS['a100'], linewidth=1.8)

apply_log2_xaxis(ax, all_nodes)
ax.set_yscale('log')
ax.set_xlabel('Number of Nodes (log₂ scale)', fontsize=12)
ax.set_ylabel('Total Time (s, log scale)',    fontsize=12)
ax.set_title('Compute Time: CPU vs Local GPU vs A100', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig('time_comparison.png', dpi=150)
plt.show()
print("Saved time_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2 — Peak Memory
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(all_nodes, cpu_mems,  label='CPU (RAM delta)',        marker=MARKERS['cpu'],
        color=COLORS['cpu'],  linewidth=1.8)
ax.plot(all_nodes, gpu_mems,  label='Local GPU (VRAM pool delta)', marker=MARKERS['gpu'],
        color=COLORS['gpu'],  linewidth=1.8)
ax.plot(all_nodes, a100_mems, label='A100 (VRAM pool delta)', marker=MARKERS['a100'],
        color=COLORS['a100'], linewidth=1.8)

apply_log2_xaxis(ax, all_nodes)
ax.set_xlabel('Number of Nodes (log₂ scale)', fontsize=12)
ax.set_ylabel('Peak Memory Used (MB)',         fontsize=12)
ax.set_title('Peak Memory: CPU vs Local GPU vs A100', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=150)
plt.show()
print("Saved memory_comparison.png")


# ---------------------------------------------------------------------------
# Tabular summary
# ---------------------------------------------------------------------------

header = (f"{'Nodes':>10}  {'CPU t(s)':>10}  {'GPU t(s)':>10}  {'A100 t(s)':>10}"
          f"  {'CPU mem(MB)':>12}  {'GPU mem(MB)':>12}  {'A100 mem(MB)':>13}")
print("\n" + "=" * len(header))
print("Time and Memory Summary (min time / max peak memory per node count)")
print("=" * len(header))
print(header)
print("-" * len(header))

for n in all_nodes:
    ct   = cpu_data.get(n,  {}).get('time',   0.0)
    gt   = gpu_data.get(n,  {}).get('time',   0.0)
    at   = a100_data.get(n, {}).get('time',   0.0)
    cm   = cpu_data.get(n,  {}).get('memory', 0.0)
    gm   = gpu_data.get(n,  {}).get('memory', 0.0)
    am   = a100_data.get(n, {}).get('memory', 0.0)
    print(f"{n:>10}  {ct:>10.3f}  {gt:>10.3f}  {at:>10.3f}"
          f"  {cm:>12.1f}  {gm:>12.1f}  {am:>13.1f}")

print("=" * len(header))


# ---------------------------------------------------------------------------
# Speedup table
# ---------------------------------------------------------------------------

print("\nSpeedups vs CPU (higher = faster than CPU):")
hdr2 = f"{'Nodes':>10}  {'GPU/CPU':>10}  {'A100/CPU':>10}"
print(hdr2)
print("-" * len(hdr2))
for n in all_nodes:
    ct = cpu_data.get(n,  {}).get('time', 0.0)
    gt = gpu_data.get(n,  {}).get('time', 0.0)
    at = a100_data.get(n, {}).get('time', 0.0)
    gpu_su  = ct / gt  if gt  > 0 else float('nan')
    a100_su = ct / at  if at  > 0 else float('nan')
    print(f"{n:>10}  {gpu_su:>10.2f}  {a100_su:>10.2f}")