"""Figures for the writeup, styled for the site's dark theme (style.css: bg #1e1e1e, text #d4d4d4)."""
import csv, json, os, sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "fig")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
REPORTS = os.path.join(WORK, "reports", "spill")   # the .sqlite exports (nsys export -t sqlite)
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID = "#1e1e1e", "#d4d4d4", "#9d9d9d", "#3c3c3c"
VRAM, SYSRAM, FIXED = "#3987e5", "#d95926", "#199e70"   # categorical slots 1-3, dark steps
REF = "#c3c2b7"
GiB = 2 ** 30


def contrast(a, b):
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


for name, col in (("blue", VRAM), ("orange", SYSRAM), ("aqua", FIXED)):
    print(f"contrast {name} vs surface: {contrast(col, SURFACE):.2f}")
print(f"contrast surface ink on blue bar: {contrast(SURFACE, VRAM):.2f}")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.family": "Verdana", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    return json.load(open(os.path.join(D, name + ".json")))


# ---------------------------------------------------------------- residency probe
LABELS = [  # key, label
    ("vram_reference", "64 MB reference buffer"),
    ("csr.data", "CSR values (float32)"),
    ("csr.indices", "CSR column indices"),
    ("csr.indptr", "CSR row pointers"),
    ("node_urns", "Urn matrix U"),
    ("edges_buffer", "Edge list buffer (dead)"),
    ("stubs", "BA stub list (dead)"),
    ("A64.data", "float64 copy of CSR values"),
    ("A64.indices", "copy of CSR column indices"),
    ("B_dense", "float64 copy of U (gathered)"),
]


def probe_json(run):
    j = load(run)
    p = dict(j["probe"]["after_polya"])
    p.update(j["probe"]["spmm_operands"])
    return {k: (p[k]["mb"] * 2 ** 20, max(p[k]["gbps"][1:])) for k, _ in LABELS}


def probe_sqlite(run, sizes):
    """Bandwidths recomputed from the probe kernels in the trace (the 71.6M harness died before
    writing its JSON). Best warm rep of the last probe of each array."""
    c = sqlite3.connect(os.path.join(REPORTS, run + ".sqlite"))
    out = {}
    for k, _ in LABELS:
        rs = c.execute("select start, end from NVTX_EVENTS where text like ? order by start",
                       (f"probe {k} #%",)).fetchall()[-3:]
        bws = []
        for s, e in rs[1:]:
            t = c.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<=?",
                          (s, e)).fetchone()[0] / 1e9
            bws.append(sizes[k] / t / 1e9)
        out[k] = (sizes[k], max(bws))
    return out


def residency(panels, fname):
    fig, axes = plt.subplots(1, len(panels), figsize=(12.5, 6.0), dpi=130, sharey=True)
    for ax, (title, data) in zip(axes, panels):
        rows = LABELS[::-1]
        bw = [data[k][1] for k, _ in rows]
        sz = [data[k][0] for k, _ in rows]
        col = [VRAM if b > 60 else SYSRAM for b in bw]
        y = range(len(rows))
        ax.barh(y, bw, color=col, height=0.62, edgecolor=SURFACE, linewidth=2)
        for i, (b, n) in enumerate(zip(bw, sz)):
            size = f"{n / GiB:.1f} GB" if n >= GiB else f"{n / 2 ** 20:.0f} MB"
            if b > 100:   # long bars: label inside the bar end so it never crosses the reference line
                ax.text(b - 4, i, f"{b:.0f} GB/s · {size}", va="center", ha="right", fontsize=9, color=SURFACE)
            else:
                ax.text(b + 4, i, f"{b:.0f} GB/s · {size}", va="center", fontsize=9, color=INK)
        ax.set_yticks(list(y), [l for _, l in rows], fontsize=10, color=INK)
        for x, lab, ha, dx in ((288, "GDDR6 spec 288", "right", -3), (15.75, "PCIe 3.0 x16 spec 15.8", "left", 4)):
            ax.axvline(x, color=REF, lw=1, ls=(0, (4, 3)))
            ax.text(x + dx, len(rows) - 0.4, lab, ha=ha, va="bottom", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 300)
        ax.set_xticks(range(0, 251, 50))
        ax.set_ylim(-0.6, len(rows) + 0.2)
        ax.set_xlabel("Measured read bandwidth (GB/s)")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=12, pad=16)
    fig.suptitle("Read bandwidth of each GPU buffer after the Pólya steps (GTX 1660 Ti)", x=0.01, ha="left",
                 fontsize=13)
    fig.tight_layout(w_pad=3, rect=(0, 0, 1, 0.97))
    fig.legend(handles=[Patch(color=VRAM, label="in VRAM"),
                        Patch(color=SYSRAM, label="in system RAM, read over PCIe")],
               loc="upper right", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.995, 0.995))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------- footprint timeline
def timeline(run):
    c = sqlite3.connect(os.path.join(REPORTS, run + ".sqlite"))
    rng = {t: (s, e) for t, s, e in c.execute(
        "select text, start, end from NVTX_EVENTS where eventType=59 and end is not null and "
        "(text in ('BA growth','CSR build') or text like 'Polya step %')")}
    t0 = rng["BA growth"][0]
    last = max(e for t, (s, e) in rng.items() if t.startswith("Polya"))
    cur, pts = 0, [(0.0, 0.0)]
    for s, b, op in c.execute("select start, bytes, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS "
                              "where memKind=2 order by start"):
        cur += b if op == 0 else -b
        if t0 <= s <= last:
            pts.append(((s - t0) / 1e9, cur / GiB))
    pts.append(((last - t0) / 1e9, cur / GiB))
    phases = {k: ((v[0] - t0) / 1e9, (v[1] - t0) / 1e9) for k, v in rng.items()}
    return pts, phases


def footprints(runs, fname, vram_gib, ceiling_gib):
    fig, axes = plt.subplots(1, len(runs), figsize=(12.5, 4.8), dpi=130, sharey=True)
    for ax, (run, label) in zip(axes, runs):
        pts, ph = timeline(run)
        xs, ys = zip(*pts)
        ax.axhspan(vram_gib, ceiling_gib, color=SYSRAM, alpha=0.10, lw=0)
        ax.axhline(vram_gib, color=SYSRAM, lw=1.2, ls=(0, (4, 3)))
        ax.axhline(ceiling_gib, color=REF, lw=1, ls=(0, (2, 2)))
        ax.step(xs, ys, where="post", color=VRAM, lw=2)
        pol = [v for k, v in ph.items() if k.startswith("Polya")]
        spans = (("growth", ph["BA growth"]), ("CSR", ph["CSR build"]),
                 ("Pólya", (min(p[0] for p in pol), max(p[1] for p in pol))))
        for name, (s, e) in spans:
            ax.axvline(s, color=GRID, lw=0.8)
            ax.text((s + e) / 2, ceiling_gib + 0.5, name, ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_title(label, loc="left", fontsize=11.5, pad=20)
        ax.set_xlim(0, xs[-1])
        ax.set_ylim(0, ceiling_gib + 2)
        ax.set_xlabel("seconds")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        i = ys.index(max(ys))
        ax.annotate(f"peak {max(ys):.1f} GB", (xs[i], ys[i]), xytext=(4, 6), textcoords="offset points",
                    ha="left" if xs[i] < xs[-1] * 0.6 else "right", fontsize=9.5, color=INK)
    axes[0].set_ylabel("CUDA memory allocated (GB)")
    # reference-line labels go in the last panel, where the curve is far from both lines
    axes[-1].text(0.98, vram_gib - 0.35, f"free VRAM at start, {vram_gib:.1f} GB",
                  transform=axes[-1].get_yaxis_transform(), va="top", ha="right", fontsize=8.5, color=INK2)
    axes[-1].text(0.98, ceiling_gib - 0.35, f"WDDM limit (VRAM + shared), {ceiling_gib:.1f} GB",
                  transform=axes[-1].get_yaxis_transform(), va="top", ha="right", fontsize=8.5, color=INK2)
    axes[-1].text(0.98, (vram_gib + ceiling_gib) / 2 - 2.2, "above the dashed line,\nallocations live in system RAM",
                  transform=axes[-1].get_yaxis_transform(), va="center", ha="right", fontsize=8.5, color=INK2)
    fig.suptitle("CUDA memory allocated during one run (Nsight Systems, CUDA memory usage)", x=0.01, ha="left",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------- WDDM residency + PCIe (admin traces)
def wddm_series(run):
    c = sqlite3.connect(os.path.join(REPORTS, run + ".sqlite"))
    S = dict(c.execute("select id, value from StringIds"))
    types = {S.get(nm): tid for tid, nm in c.execute("select typeId, nameId from GENERIC_EVENT_TYPES")}
    pid = (c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0] >> 24) & 0xFFFFFF
    rng = {t: (s, e) for t, s, e in c.execute(
        "select coalesce(text,''), start, end from NVTX_EVENTS where eventType=59 and end is not null and "
        "(text in ('BA growth','CSR build') or text like 'Polya step %')")}
    t0 = rng["BA growth"][0]
    t1 = max(e for k, (s, e) in rng.items() if k.startswith("Polya"))

    def etw(name, seg=None):
        pts = []
        for ts, d in c.execute("select timestamp, data from ETW_EVENTS where typeId=? order by timestamp", (types[name],)):
            d = json.loads(d)
            if int(d.get("ProcessId", -1)) == pid and (seg is None or d.get("MemorySegmentGroup") == seg):
                pts.append(((ts - t0) / 1e9, int(d["Commitment"]) / GiB))
        return pts

    vram = etw("VidMmProcessCommitmentChange", "Local")       # resident in VRAM
    dem = etw("VidMmProcessDemotedCommitmentChange")           # demoted to system RAM
    mid = c.execute("select metricId from TARGET_INFO_GPU_METRICS where metricName='PCIe RX Throughput [Throughput %]'").fetchone()[0]
    bins = {}
    for ts, v in c.execute("select timestamp, value from GPU_METRICS where metricId=? and timestamp between ? and ?",
                           (mid, t0, t1)):
        b = int((ts - t0) / 1e8)                                 # 100 ms bins
        s, n = bins.get(b, (0.0, 0))
        bins[b] = (s + v, n + 1)
    pcie = sorted((b / 10, s / n) for b, (s, n) in bins.items())
    phases = {k: ((s - t0) / 1e9, (e - t0) / 1e9) for k, (s, e) in rng.items()}
    return vram, dem, pcie, phases, (t1 - t0) / 1e9


def wddm_figure(runs, fname):
    fig, axes = plt.subplots(2, len(runs), figsize=(12.5, 6.4), dpi=130, sharey="row",
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for col, (run, label) in enumerate(runs):
        vram, dem, pcie, ph, end = wddm_series(run)
        grid = [i * end / 2000 for i in range(2001)]

        def at(s, t):
            v = [b for ts, b in s if ts <= t]
            return v[-1] if v else 0.0
        v = [at(vram, t) for t in grid]
        d = [at(dem, t) for t in grid]
        top, bot = axes[0][col], axes[1][col]
        top.fill_between(grid, 0, v, step="post", color=VRAM, lw=0, label="in VRAM")
        top.fill_between(grid, v, [a + b for a, b in zip(v, d)], step="post", color=SYSRAM, lw=0,
                         label="demoted to system RAM")
        bot.plot([p[0] for p in pcie], [p[1] for p in pcie], color=SYSRAM, lw=1.5)
        pol = [v for k, v in ph.items() if k.startswith("Polya")]
        p0, p1 = min(p[0] for p in pol), max(p[1] for p in pol)
        for name, (s, e) in (("growth", ph["BA growth"]), ("CSR build", ph["CSR build"]), ("Pólya", (p0, p1))):
            for ax in (top, bot):
                ax.axvline(s, color=GRID, lw=0.8)
            top.text((s + e) / 2, 21.3, name, ha="center", va="bottom", fontsize=8.5, color=INK2)
        # the read-bandwidth probes run between the CSR build and the first step: measurement, not simulation
        for ax in (top, bot):
            ax.axvspan(ph["CSR build"][1], p0, color=INK2, alpha=0.12, lw=0)
        bot.text((ph["CSR build"][1] + p0) / 2, 97, "probes", ha="center", va="top", fontsize=8, color=INK2)
        top.set_title(label, loc="left", fontsize=11.5, pad=36)
        top.text(0, 1.10, f"during the steps: {at(vram, end):.1f} GB in VRAM, {at(dem, end):.1f} GB in system RAM",
                 transform=top.transAxes, fontsize=9, color=INK2, va="bottom")
        for ax in (top, bot):
            ax.set_xlim(0, end)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        top.set_ylim(0, 21)
        bot.set_ylim(0, 100)
        bot.set_xlabel("seconds")
    axes[0][0].set_ylabel("GPU memory (GB)")
    axes[1][0].set_ylabel("PCIe reads\n(% of link peak)")
    fig.suptitle("Where Windows kept the simulation's memory, and the PCIe reads it caused (WDDM trace + GPU Metrics)",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles=[Patch(color=VRAM, label="in VRAM"), Patch(color=SYSRAM, label="demoted to system RAM")],
               loc="upper right", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.995, 0.955))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------- per-step time vs size
def bench(csv_name):
    rows = list(csv.DictReader(open(os.path.join(REPO, "performance_analysis", "raw", csv_name))))
    keep = {}
    for r in rows:   # the compute-time chart on the site uses these configs for the two duplicated sizes
        n = int(r["total_nodes"])
        if n == 17895744 and not r["batch_size"].startswith("4;"):
            continue
        if n == 35791424 and not r["batch_size"].startswith("8;"):
            continue
        keep[n] = float(r["polya_time"]) / int(r["polya_steps"])
    return sorted(keep.items())


def per_step(fname, fixed_runs):
    cpu, gpu = bench("benchmark_results_CPU_BIG_FINAL.csv"), bench("benchmark_results_GPU_BIG_FINAL.csv")
    fx = []
    for run in fixed_runs:
        j = load(run)
        st = j["steps"][1:]                      # step 0 carries one-off JIT + first allocations
        fx.append((j["nodes"], sum(st) / len(st)))
    fx.sort()
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=130)
    lo = 1_000_000
    # end labels: CPU and benchmarked GPU finish 1% apart, so nudge them vertically
    for pts, col, lab, mk, dy, pre in ((cpu, VRAM, "Ryzen 3 3100 CPU", "o", 9, "CPU "),
                                       (gpu, SYSRAM, "GTX 1660 Ti, as benchmarked", "s", -3, "GPU "),
                                       (fx, FIXED, "GTX 1660 Ti, float32 SpMM + dead buffers freed", "^", 0, "")):
        pts = [p for p in pts if p[0] >= lo]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=7, mec=SURFACE, mew=1.5, label=lab)
        ax.annotate(f"{pre}{ys[-1]:.1f} s", (xs[-1], ys[-1]), xytext=(10, dy), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK)
    # clean re-runs of the unmodified code: in the spill regime the step time depends on which
    # buffers the driver leaves in system RAM, so it varies from run to run
    rr = [(load(r)["nodes"], sum(load(r)["steps"][1:]) / len(load(r)["steps"][1:])) for r in ("n18M", "n36M", "n72M")]
    ax.plot(*zip(*rr), ls="none", marker="s", ms=8, mfc="none", mec=SYSRAM, mew=1.8,
            label="GTX 1660 Ti, unmodified code, re-measured")
    ax.annotate(f"{rr[-1][1]:.1f} s", rr[-1], xytext=(10, -15), textcoords="offset points",
                va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [n for n, _ in cpu if n >= lo]
    ax.set_xticks(ticks, [f"{n / 1e6:.1f} M" for n in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.25, ticks[-1] * 1.6)
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("Nodes (log scale)")
    ax.set_ylabel("Seconds per Pólya step (log scale)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title("Time per Pólya step, m = 3 Barabási–Albert network", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)
    for (n, c), (_, g) in zip(cpu, gpu):
        f = dict(fx).get(n)
        print(f"  {n:>11,}  cpu {c:7.3f}  gpu {g:7.3f}  fixed {f if f is None else round(f, 3)}")


if __name__ == "__main__":
    residency([("35.8 M nodes", probe_json("n36M")), ("71.6 M nodes", probe_json("n72M"))],
              "capstone-residency.png")
    footprints([("n18M", "17.9 M nodes"), ("n36M", "35.8 M nodes"), ("n72M", "71.6 M nodes")],
               "capstone-gpu-footprint.png", 5100 / 1024, 22324 / 1024)
    per_step("capstone-step-time.png", ["f1M", "f2M", "f4M", "f9M", "f18M", "n36M_fix", "n72M_fix"])
    wddm_figure([("admin/a72", "71.6 M nodes, current code"), ("admin/a72f", "71.6 M nodes, both fixes")],
                "capstone-wddm.png")
