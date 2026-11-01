"""Summarize an admin nsys export: WDDM VRAM/shared usage, driver paging, GPU Metrics PCIe/DRAM per range."""
import json, os, sqlite3, sys
from collections import defaultdict

db = sys.argv[1]
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))

gpid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]
pid = (gpid >> 24) & 0xFFFFFF

ranges = [(t, s, e) for t, s, e in c.execute(
    "select coalesce(text, ''), start, end from NVTX_EVENTS where eventType=59 and end is not null order by start")]
phases = [r for r in ranges if r[0] in ("BA growth", "CSR build") or r[0].startswith("Polya step")]
probes = [r for r in ranges if r[0].startswith("probe ")]
t0 = phases[0][1]

# ---- GPU Metrics
mid = {name: i for i, name in c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
want = {"pcie_rx": "PCIe RX Throughput [Throughput %]", "pcie_tx": "PCIe TX Throughput [Throughput %]",
        "dram_rd": "DRAM Read Bandwidth [Throughput %]", "dram_wr": "DRAM Write Bandwidth [Throughput %]",
        "sm_active": "SMs Active [Throughput %]"}
series = {}
for key, name in want.items():
    series[key] = c.execute("select timestamp, value from GPU_METRICS where metricId=? order by timestamp",
                            (mid[name],)).fetchall()


def mean_in(key, s, e):
    import bisect
    ts = [x[0] for x in series[key]]
    i, j = bisect.bisect_left(ts, s), bisect.bisect_right(ts, e)
    v = [x[1] for x in series[key][i:j]]
    return sum(v) / len(v) if v else float("nan")


# ---- WDDM usage / budget for our process
types = {S.get(nm): tid for tid, nm in c.execute("select typeId, nameId from GENERIC_EVENT_TYPES")}
usage = defaultdict(list)       # (segment) -> [(t, bytes)]
budget = defaultdict(list)
demoted = []
for name, store, field in (("VidMmProcessUsageChange", usage, "NewUsage"),
                           ("VidMmProcessBudgetChange", budget, "NewBudget")):
    for ts, data in c.execute("select timestamp, data from ETW_EVENTS where typeId=? order by timestamp",
                              (types[name],)):
        d = json.loads(data)
        if int(d["ProcessId"]) == pid:
            store[d["MemorySegmentGroup"]].append((ts, int(d[field])))
if "VidMmProcessDemotedCommitmentChange" in types:
    for ts, data in c.execute("select timestamp, data from ETW_EVENTS where typeId=? order by timestamp",
                              (types["VidMmProcessDemotedCommitmentChange"],)):
        d = json.loads(data)
        if int(d.get("ProcessId", -1)) == pid:
            demoted.append((ts, d))


def last_before(store, seg, t):
    v = [b for ts, b in store.get(seg, []) if ts <= t]
    return v[-1] if v else 0


def max_in(store, seg, s, e):
    v = [b for ts, b in store.get(seg, []) if s <= ts <= e]
    return max(v + [last_before(store, seg, s)])


# ---- driver paging (all processes; our process dominates while it runs)
xfer = c.execute("select start, size, memoryTransferType from MEMORY_TRANSFER_EVENTS").fetchall()
TT = {0: "unknown", 1: "VRAM->sysRAM", 2: "sysRAM->VRAM", 5: "evict", 6: "restore", 7: "discard"}

GiB = 2 ** 30
print(f"pid {pid}; local budget {last_before(budget, 'Local', phases[-1][2]) / GiB:.2f} GiB, "
      f"non-local budget {last_before(budget, 'NonLocal', phases[-1][2]) / GiB:.2f} GiB; demotion events {len(demoted)}")
out = {"pid": pid, "ranges": [], "probes": []}
for name, s, e in phases:
    tr = defaultdict(int)
    for ts, size, kind in xfer:
        if s <= ts <= e:
            tr[TT.get(kind, kind)] += size
    r = {"name": name, "t0": (s - t0) / 1e9, "t1": (e - t0) / 1e9,
         "local_max_gib": max_in(usage, "Local", s, e) / GiB, "nonlocal_max_gib": max_in(usage, "NonLocal", s, e) / GiB,
         "local_end_gib": last_before(usage, "Local", e) / GiB, "nonlocal_end_gib": last_before(usage, "NonLocal", e) / GiB,
         "paged_gib": {k: v / GiB for k, v in tr.items()},
         **{k: mean_in(k, s, e) for k in want}}
    out["ranges"].append(r)
    print(f"{name:<14} {r['t1'] - r['t0']:7.2f}s  VRAM {r['local_end_gib']:5.2f} GiB  shared {r['nonlocal_end_gib']:5.2f} GiB  "
          f"PCIe rx {r['pcie_rx']:5.1f}% tx {r['pcie_tx']:5.1f}%  DRAM rd {r['dram_rd']:5.1f}%  SM {r['sm_active']:5.1f}%  "
          f"paged {', '.join(f'{k} {v:.2f}' for k, v in r['paged_gib'].items() if v > 0.001)}")
for name, s, e in probes:
    out["probes"].append({"name": name, "s": s, "e": e, "pcie_rx": mean_in("pcie_rx", s, e),
                          "dram_rd": mean_in("dram_rd", s, e)})

# timelines for figures (decimated)
step = max(1, len(series["pcie_rx"]) // 6000)
out["timeline"] = {k: [((t - t0) / 1e9, v) for t, v in series[k][::step]] for k in series}
out["usage"] = {seg: [((t - t0) / 1e9, b / GiB) for t, b in v] for seg, v in usage.items()}
out["budget"] = {seg: [((t - t0) / 1e9, b / GiB) for t, b in v] for seg, v in budget.items()}
json.dump(out, open(os.path.splitext(db)[0] + "_admin.json", "w"))
