"""Summarizes an engine trace (sqlite export): per step, GPU busy time by kernel, host time waiting for block data,
and the mean PCIe receive / DRAM read throughput over the step."""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
q = lambda s, *a: db.execute(s, a).fetchall()
strs = dict(q("select id, value from StringIds"))
steps = [(s, e, t) for s, e, t in q("select start, end, coalesce(text, '') from NVTX_EVENTS where text like 'step %' order by start")]
waits = q("select start, end from NVTX_EVENTS where text = 'wait for block data'")
kern = q("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")
tables = {r[0] for r in q("select name from sqlite_master where type='table'")}
metrics = {}
if "GPU_METRICS" in tables and "TARGET_INFO_GPU_METRICS" in tables:
    names = dict(q("select metricId, metricName from TARGET_INFO_GPU_METRICS"))
    want = {i: nm for i, nm in names.items() if nm.startswith(("PCIe RX Throughput", "DRAM Read Bandwidth", "SMs Active"))}
    for i, nm in want.items():
        metrics[nm] = q("select timestamp, value from GPU_METRICS where metricId = ?", i)
for s, e, t in steps:
    # the step's GPU work ends at the last kernel that starts before the next step's first launch; use the range
    ks = [(a, b, strs.get(n, n)) for a, b, n in kern if s <= a < e + 2e9]
    busy = {}
    for a, b, n in kern:
        if s <= a < e:
            busy[n] = busy.get(strs.get(n, n), 0) + (b - a)
    w = sum(min(b, e) - max(a, s) for a, b in waits if a < e and b > s)
    line = f"{t}: host range {(e - s) / 1e6:.0f} ms, waiting for block data {w / 1e6:.0f} ms"
    tot = sum(busy.values())
    line += f", kernels launched in range {tot / 1e6:.0f} ms"
    for nm, pts in metrics.items():
        v = [val for ts, val in pts if s <= ts < e]
        if v:
            line += f", {nm} {sum(v) / len(v):.0f}%"
    print(line)
names = {}
for a, b, n in kern:
    k = strs.get(n, n)
    names[k] = names.get(k, 0) + (b - a)
print("kernel totals (ms):", {k: round(v / 1e6, 1) for k, v in sorted(names.items(), key=lambda x: -x[1])[:8]})
