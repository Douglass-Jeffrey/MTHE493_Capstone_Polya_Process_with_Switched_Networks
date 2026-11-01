"""Summarize an nsys SQLite export: per-NVTX-range kernel breakdown, GPU idle time, allocations."""
import json, sqlite3, sys
from collections import defaultdict

db = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))

ranges = [(t, s, e) for t, s, e in c.execute(
    "select coalesce(text, ''), start, end from NVTX_EVENTS where end is not null and eventType=59")]
ranges = [r for r in ranges if r[0] in ("BA growth", "CSR build") or r[0].startswith("Polya step")]
kern = c.execute("select start, end, shortName, gridX*gridY*gridZ, blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
cpy = c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
mem = c.execute("select start, bytes, memKind, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS where memKind in (2,4) order by start").fetchall()

managed = sum(1 for m in mem if m[2] == 4)
print(f"managed (cudaMallocManaged) allocations: {managed}")

# device-memory footprint over time (what CuPy has cudaMalloc'd, incl. cached pool blocks)
cur = peak = 0
foot = []
for s, b, k, op in mem:
    cur += b if op == 0 else -b
    peak = max(peak, cur)
    foot.append((s, cur))

out = {"ranges": []}
for name, s, e in ranges:
    wall = (e - s) / 1e9
    ks = [k for k in kern if s <= k[0] <= e]
    busy = sum(k[1] - k[0] for k in ks) / 1e9
    by = defaultdict(lambda: [0.0, 0])
    for k in ks:
        by[S.get(k[2], k[2])][0] += (k[1] - k[0]) / 1e9
        by[S.get(k[2], k[2])][1] += 1
    cs = [x for x in cpy if s <= x[0] <= e]
    alloc = sum(b for t, b, kk, op in mem if s <= t <= e and op == 0)
    fp = [v for t, v in foot if t <= e]
    footprint = max([v for t, v in foot if s <= t <= e] + ([fp[-1]] if fp else [0]))
    r = {"name": name, "wall_s": wall, "kernel_s": busy, "idle_s": wall - busy, "n_kernels": len(ks),
         "memcpy": {str(k): [sum(x[2] for x in cs if x[3] == k) / 1e9, sum(x[1] - x[0] for x in cs if x[3] == k) / 1e9]
                    for k in set(x[3] for x in cs)},
         "new_cudamalloc_gb": alloc / 1e9, "device_footprint_gb": footprint / 1e9,
         "kernels": sorted(([n, v[0], v[1]] for n, v in by.items()), key=lambda x: -x[1])}
    out["ranges"].append(r)
    print(f"\n== {name}: wall {wall:.3f}s  kernels {busy:.3f}s ({len(ks)})  gpu-idle {wall - busy:.3f}s  "
          f"new cudaMalloc {alloc / 1e9:.2f} GB  footprint {footprint / 1e9:.2f} GB")
    for kk, (gb, t) in r["memcpy"].items():
        print(f"   memcpy kind {kk}: {gb:.2f} GB in {t:.3f}s")
    for n, t, cnt in r["kernels"][:top]:
        print(f"   {t:9.4f}s  x{cnt:<4} {n[:90]}")
print(f"\npeak device footprint: {peak / 1e9:.2f} GB")
out["peak_footprint_gb"] = peak / 1e9
out["footprint"] = foot[:: max(1, len(foot) // 4000)]
json.dump(out, open(db.replace(".sqlite", "_analysis.json"), "w"))
