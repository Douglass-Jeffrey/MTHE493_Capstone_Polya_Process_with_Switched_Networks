"""Times the out-of-core engine's step on a split-format graph; appends one JSON line to results.jsonl."""
import argparse
import json
import os
import time

import cupy as cp
import numpy as np
import psutil
from cupy.cuda import nvtx

from capstone_optimized.scale.engine import Engine

S = r"C:\Users\dougl\AppData\Local\Temp\claude\c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks\7bf0ea5c-bb18-4a6c-8e60-f701292fffbe\scratchpad"
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="split-format directory (absolute, or a name under data/)")
ap.add_argument("--placement", default="auto")
ap.add_argument("--block-mb", type=float, default=None)
ap.add_argument("--block-rows", type=int, default=1 << 22)
ap.add_argument("--pinned-gb", type=float, default=8.0)
ap.add_argument("--ram-gb", type=float, default=0.0)
ap.add_argument("--slots", type=int, default=6)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--lanes", type=int, default=4)
ap.add_argument("--l2", type=int, default=None)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--cpu-blocks", type=int, default=0)
ap.add_argument("--cpu-threads", type=int, default=8)
ap.add_argument("--auto-budget", action="store_true",
                help="size the pinned and pageable-RAM tiers from the RAM available now")
ap.add_argument("--reserve-gb", type=float, default=3.0, help="RAM left to the OS and other programs")
ap.add_argument("--pin-limit-gb", type=float, default=14.5, help="what cudaHostAlloc allows in total (15 GB measured)")
ap.add_argument("--memory", type=int, default=0, help="finite memory M (0 = off)")
ap.add_argument("--ring-file", default=None)
ap.add_argument("--push-k", type=int, default=0, help="push rows [init, K); -1 = as many as fit in VRAM")
ap.add_argument("--vram-margin", type=float, default=0.4)
ap.add_argument("--tag", default="")
args = ap.parse_args()
d = args.dir if os.path.isabs(args.dir) else os.path.join(S, "data", args.dir)

proc = psutil.Process()
if args.auto_budget:
    from capstone_optimized.scale import split_format as _sf
    _n = _sf.read_meta(d)["n"]
    avail = psutil.virtual_memory().available / 2 ** 30
    slots_gb = args.slots * ((args.block_mb or 128) + 2) / 1024
    fixed = _n / 2 ** 30 + slots_gb + 1.5 + args.reserve_gb
    args.pinned_gb = max(0.0, min(args.pin_limit_gb - _n / 2 ** 30 - slots_gb, avail - fixed))
    args.ram_gb = max(0.0, avail - fixed - args.pinned_gb)
    print(f"auto budget: available {avail:.1f} GiB -> pinned {args.pinned_gb:.1f} GiB, ram {args.ram_gb:.1f} GiB",
          flush=True)
t0 = time.time()
eng = Engine(d, placement=args.placement, block_rows=args.block_rows, block_mb=args.block_mb, pinned_budget_gb=args.pinned_gb,
             ram_budget_gb=args.ram_gb, slots=args.slots, threads=args.threads, lanes=args.lanes, seed=1,
             l2_fetch=args.l2, push_k=args.push_k, vram_margin_gb=args.vram_margin)
if args.cpu_blocks:
    eng.enable_cpu(args.cpu_blocks, threads=args.cpu_threads)
if args.memory:
    eng.enable_memory(args.memory, ring_path=args.ring_file)
load = time.time() - t0
free_after = cp.cuda.runtime.memGetInfo()[0]
for _ in range(args.warmup):
    eng.step()
cp.cuda.Device().synchronize()
b0 = eng.feeder.bytes_read
nvtx.RangePush("timed steps")
per, wall = eng.run(args.steps)
nvtx.RangePop()
nbytes = (eng.feeder.bytes_read - b0) / args.steps
rec = {"tag": args.tag, "dir": os.path.basename(d), "n": eng.n, "placement": eng.placement,
       "block_rows": args.block_rows, "block_mb": args.block_mb, "blocks": len(eng.blocks),
       "pinned_blocks": sum(b["where"] == "pinned" for b in eng.blocks),
       "ram_blocks": sum(b["where"] == "ram" for b in eng.blocks),
       "stream_blocks": sum(b["where"] == "stream" for b in eng.blocks),
       "pinned_GiB": round(eng.pinned_bytes / 2 ** 30, 2), "ram_GiB": round(eng.ram_bytes / 2 ** 30, 2),
       "slots": args.slots, "threads": args.threads, "lanes": args.lanes, "l2": args.l2,
       "load_s": round(load, 1), "step_s": [round(x, 4) for x in per], "mean_step_s": round(float(np.mean(per)), 4),
       "wall_per_step_s": round(wall / args.steps, 4), "disk_GB_per_step": round(nbytes / 1e9, 2),
       "vram_free_GiB": round(free_after / 2 ** 30, 2), "rss_GiB": round(proc.memory_info().rss / 2 ** 30, 2),
       "push_K": eng.K, "memory": args.memory, "overflow": eng.overflowed(), "cpu_blocks": args.cpu_blocks,
       "cpu_apply_draw_s": ([round(float(np.mean([a for a, b in eng.cpu_times[-args.steps:]])), 4),
                             round(float(np.mean([b for a, b in eng.cpu_times[-args.steps:]])), 4)]
                            if args.cpu_blocks else None)}
print(json.dumps(rec), flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")
eng.close()
