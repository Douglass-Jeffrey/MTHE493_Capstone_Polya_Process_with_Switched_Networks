"""
Workstream A: DRAM bytes read by a kernel, from Nsight Systems' GPU metrics (Nsight Compute is not installed).
Profiles bench_quick.py, then integrates "DRAM Read Bandwidth [Throughput %]" over each matching kernel.

  python nsys_dram.py n71582784 u16_phx 64      (variant, L2 fetch granularity)
"""
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
PEAK = 288e9                          # 6001 MHz x 2 x 192 bit / 8
cache, variant, gran = sys.argv[1], sys.argv[2], sys.argv[3]
kname = sys.argv[4] if len(sys.argv) > 4 else "draw_"
out = os.path.join(HERE, "rep", f"{cache}_{variant}_g{gran}")
os.makedirs(os.path.dirname(out), exist_ok=True)
subprocess.check_call([NSYS, "profile", "--force-overwrite=true", "-t", "cuda", "--gpu-metrics-devices=0",
                       "--gpu-metrics-frequency=20000", "-o", out, sys.executable,
                       os.path.join(HERE, "bench_quick.py"), cache, "--variants", variant, "--gran", gran,
                       "--steps", "3", "--warmup", "0"], stdout=subprocess.DEVNULL)
subprocess.check_call([NSYS, "export", "--force-overwrite=true", "--type", "sqlite", "-o", out + ".sqlite",
                       out + ".nsys-rep"], stdout=subprocess.DEVNULL)
c = sqlite3.connect(out + ".sqlite")
mid = {nm: i for i, nm in c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
rows = np.array(c.execute("select timestamp, value from GPU_METRICS where metricId=? order by timestamp",
                          (mid["DRAM Read Bandwidth [Throughput %]"],)).fetchall(), dtype=float)
kern = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s "
                 "on k.shortName = s.id order by k.start").fetchall()
res = []
for s, e, nm in kern:
    if kname not in nm or e - s < 5e6:
        continue
    sel = (rows[:, 0] >= s) & (rows[:, 0] <= e)
    pct = rows[sel, 1].mean()
    res.append({"kernel": nm, "ms": round((e - s) / 1e6, 1), "dram_rd_pct": round(pct, 1),
                "dram_rd_gb": round(pct / 100 * PEAK * (e - s) / 1e9 / 1e9, 2)})
print(json.dumps({"cache": cache, "variant": variant, "gran": int(gran), "kernels": res[-3:]}))
