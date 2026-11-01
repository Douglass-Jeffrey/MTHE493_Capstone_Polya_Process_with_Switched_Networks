"""Per-step time vs nodes: CPU, GPU as benchmarked, GPU with the float32/free fixes, GPU with the fused kernel
(repo benchmark re-run), and the fused kernel on a BFS-ordered graph. Same style as prof/figures.py."""
import csv, json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(os.path.dirname(D), "profiling", "spill")
sys.path.insert(0, PROF)
import figures as F          # rcParams + palette of the site figures
import matplotlib.pyplot as plt

REPO = F.REPO
PURPLE = "#a970e0"


def bench(path):
    keep = {}
    for r in csv.DictReader(open(path)):
        n = int(r["total_nodes"])
        if n == 17895744 and not r["batch_size"].startswith("4;"):
            continue
        if n == 35791424 and not r["batch_size"].startswith("8;"):
            continue
        keep[n] = float(r["polya_time"]) / int(r["polya_steps"])
    return sorted(keep.items())


raw = os.path.join(REPO, "performance_analysis", "raw")
cpu = bench(os.path.join(raw, "benchmark_results_CPU_BIG_FINAL.csv"))
gpu = bench(os.path.join(raw, "benchmark_results_GPU_BIG_FINAL.csv"))
new = bench(os.path.join(D, "bench_new", "benchmark_results.csv"))
b143 = json.load(open(os.path.join(os.path.dirname(D), "prof2", "rep", "b143.json")))
new.append((b143["nodes"], sorted(b143["steps"][1:])[len(b143["steps"][1:]) // 2]))   # zero-copy indices
fx = []
for run in ["f1M", "f2M", "f4M", "f9M", "f18M", "n36M_fix", "n72M_fix"]:
    j = json.load(open(os.path.join(PROF, run + ".json")))
    st = j["steps"][1:]
    fx.append((j["nodes"], sum(st) / len(st)))
bfs = {}
for line in open(os.path.join(D, "results.jsonl")):
    r = json.loads(line)
    if r["mode"] == "resident" and r["lanes"] == 4 and r["cache"].endswith("_bfs") and not r["tag"]:
        bfs[r["n"]] = r["median_s"]
bfs = sorted(bfs.items())

fig, ax = plt.subplots(figsize=(11, 5.8), dpi=130)
lo = 1_000_000
series = [(cpu, F.VRAM, "Ryzen 3 3100 CPU", "o", "-", 8, "CPU "),
          (gpu, F.SYSRAM, "GTX 1660 Ti, original code", "s", "-", -6, "GPU "),
          (sorted(fx), F.REF, "GTX 1660 Ti, float32 product + dead buffers freed", "^", (0, (4, 3)), 0, ""),
          (new, F.FIXED, "GTX 1660 Ti, fused kernel (143.2 M: indices in pinned system RAM)", "D", "-", 0, ""),
          (bfs, PURPLE, "GTX 1660 Ti, fused kernel, breadth-first node order (not integrated)", "v", (0, (1.5, 2)), 0, "")]
for pts, col, lab, mk, ls, dy, pre in series:
    pts = [p for p in pts if p[0] >= lo]
    if not pts:
        continue
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=2, ls=ls, marker=mk, ms=7, mec=F.SURFACE, mew=1.5, label=lab)
    ax.annotate(pre + (f"{ys[-1] * 1e3:.0f} ms" if ys[-1] < 1 else f"{ys[-1]:.1f} s"), (xs[-1], ys[-1]),
                xytext=(10, dy), textcoords="offset points", va="center", fontsize=9.5, color=F.INK)
n72, t72 = [p for p in new if p[0] == 71582784][0]
ax.annotate(f"{t72 * 1e3:.0f} ms", (n72, t72), xytext=(-10, 10), textcoords="offset points", ha="right",
            fontsize=9.5, color=F.INK)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ticks = [n for n, _ in cpu if n >= lo] + [b143["nodes"]]
ax.set_xticks(ticks, [f"{n / 1e6:.1f} M" for n in ticks])
ax.minorticks_off()
ax.set_xlim(ticks[0] / 1.25, ticks[-1] * 1.7)
ax.grid(True, which="major")
ax.set_axisbelow(True)
ax.set_xlabel("Nodes (log scale)")
ax.set_ylabel("Seconds per Pólya step (log scale)")
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
ax.set_title("Time per Pólya step, m = 3 Barabási–Albert network", loc="left", pad=12)
fig.tight_layout()
os.makedirs(os.path.join(D, "fig"), exist_ok=True)
fig.savefig(os.path.join(D, "fig", "capstone-step-time-fused.png"))
print("wrote fig/capstone-step-time-fused.png")
print(f"{'nodes':>12} {'cpu':>8} {'gpu':>8} {'fixed':>8} {'fused':>8} {'bfs':>8}")
d = {k: dict(v) for k, v in (("cpu", cpu), ("gpu", gpu), ("fx", fx), ("new", new), ("bfs", bfs))}
for n in sorted(set(d["cpu"]) | set(d["new"])):
    print(f"{n:>12,} " + " ".join(f"{d[k].get(n, float('nan')):8.4f}" for k in ("cpu", "gpu", "fx", "new", "bfs")))
