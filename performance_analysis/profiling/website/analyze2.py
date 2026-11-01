"""Per-phase summary of an Nsight Systems sqlite export: NVTX phases, kernels per Polya step, GPU Metrics
means (PCIe, DRAM, SM), CUDA allocation total, and the WDDM VRAM / system-RAM usage of the process.
Usage: analyze2.py rep/<name>.sqlite [--json out.json]"""
import bisect, json, sqlite3, sys
from collections import defaultdict

path = sys.argv[1]
c = sqlite3.connect(path)
S = dict(c.execute("select id, value from StringIds"))
out = {"report": path, "phases": {}, "steps": []}

gpid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]
pid = (gpid >> 24) & 0xFFFFFF
ranges = [(t or S.get(tid, ""), s, e) for t, tid, s, e in
          c.execute("select text, textId, start, end from NVTX_EVENTS where eventType=59 and end is not null order by start")]

# GPU metrics
mid = {n: i for i, n in c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
want = {"pcie_rx": "PCIe RX Throughput [Throughput %]", "pcie_tx": "PCIe TX Throughput [Throughput %]",
        "dram_rd": "DRAM Read Bandwidth [Throughput %]", "sm_active": "SMs Active [Throughput %]",
        "sm_issue": "SM Issue [Throughput %]"}
series = {}
for k, n in want.items():
    if n in mid:
        rows = c.execute("select timestamp, value from GPU_METRICS where metricId=? order by timestamp", (mid[n],)).fetchall()
        series[k] = ([r[0] for r in rows], [r[1] for r in rows])


def mean_in(k, s, e):
    if k not in series:
        return None
    ts, vs = series[k]
    i, j = bisect.bisect_left(ts, s), bisect.bisect_right(ts, e)
    return round(sum(vs[i:j]) / (j - i), 1) if j > i else None


# CUDA allocation running total (memoryOperationType 0 = alloc, 1 = free)
ev = c.execute("select start, bytes, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start").fetchall()
mem_t, mem_v, tot = [], [], 0
for t, b, op in ev:
    tot += b if op == 0 else -b
    mem_t.append(t)
    mem_v.append(tot)


def mem_peak(s, e):
    i, j = bisect.bisect_left(mem_t, s), bisect.bisect_right(mem_t, e)
    v = mem_v[i:j] + ([mem_v[i - 1]] if i > 0 else [])
    return round(max(v) / 1e9, 2) if v else None


# WDDM usage per segment group ("Local" = VRAM, "NonLocal" = system RAM) for this process
types = {S.get(nm): tid for tid, nm in c.execute("select typeId, nameId from GENERIC_EVENT_TYPES")} \
    if c.execute("select name from sqlite_master where name='GENERIC_EVENT_TYPES'").fetchone() else {}
usage = defaultdict(list)
if "VidMmProcessUsageChange" in types:
    for ts, data in c.execute("select timestamp, data from ETW_EVENTS where typeId=? order by timestamp",
                              (types["VidMmProcessUsageChange"],)):
        d = json.loads(data)
        if int(d["ProcessId"]) == pid:
            usage[str(d["MemorySegmentGroup"])].append((ts, int(d["NewUsage"])))


def wddm_max(seg, s, e):
    v = [b for ts, b in usage.get(seg, []) if s <= ts <= e] + [b for ts, b in usage.get(seg, []) if ts <= s][-1:]
    return round(max(v) / 1e9, 2) if v else None


kern = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
cpy = c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
for name, s, e in ranges:
    if name in ("BA growth", "CSR build"):
        out["phases"][name] = {"s": round((e - s) / 1e9, 2), "alloc_peak_gb": mem_peak(s, e),
                               "vram_gb": wddm_max("Local", s, e), "sysram_gb": wddm_max("NonLocal", s, e)}
    if not (name.startswith("Polya step") or " step " in name):
        continue
    ks = defaultdict(float)
    for ks_, ke_, sn in kern:
        if ks_ >= s and ke_ <= e:
            ks[S.get(sn, sn)] += (ke_ - ks_) / 1e9
    cp_ = defaultdict(float)
    for cs, ce, b, kind in cpy:
        if cs >= s and ce <= e:
            cp_[kind] += b / 1e9
    out["steps"].append({"name": name, "s": round((e - s) / 1e9, 4),
                         "kernels": {k: round(v, 4) for k, v in sorted(ks.items(), key=lambda x: -x[1])[:4]},
                         "memcpy_gb": {k: round(v, 3) for k, v in cp_.items()},
                         **{k: mean_in(k, s, e) for k in want},
                         "alloc_peak_gb": mem_peak(s, e), "vram_gb": wddm_max("Local", s, e), "sysram_gb": wddm_max("NonLocal", s, e)})
out["alloc_peak_total_gb"] = round(max(mem_v) / 1e9, 2) if mem_v else None
for k, v in out["phases"].items():
    print(f"{k:12s} {v}")
for st in out["steps"]:
    print(f"{st['name']:24s} {st['s']:8.4f}s  rx {st['pcie_rx']}% tx {st['pcie_tx']}% dram {st['dram_rd']}% "
          f"sm {st['sm_active']}/{st['sm_issue']}% alloc {st['alloc_peak_gb']} GB vram {st['vram_gb']} sys {st['sysram_gb']} "
          f"memcpy {st['memcpy_gb']} {st['kernels']}")
print("alloc peak", out["alloc_peak_total_gb"], "GB")
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
