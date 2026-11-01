"""Runs bench_engine.py once per configuration line (each in a fresh process), for use under one GPU lock.
    python sweep.py <configs.txt>      one line per run: bench_engine.py arguments; # comments allowed
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if not line:
        continue
    print(f"=== {line}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(here, "bench_engine.py")] + line.split(),
                       capture_output=True, text=True)
    out = [ln for ln in r.stdout.splitlines() if ln.startswith(("{", "engine:", "auto budget"))]
    print("\n".join(out) if r.returncode == 0 else f"FAILED ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}",
          flush=True)
