"""
Workstream A: finite memory. The repo keeps the last M draws as an (M, n) int32 array (288 B per node at
M = 72: 5.2 GB at 17.9M nodes, 20.6 GB at 71.6M) and updates it with fancy indexing. The variant here keeps
one bit per node and step (9 B per node at M = 72) and does the urn update, the expiry and the ring write in
one kernel (quick_kernels.apply_u16_mem).

  python bench_mem.py n1118544 --verify          urns identical to the repo's Polya_Process with memory
  python bench_mem.py n17895744 --mode repo      time the repo's step with memory
  python bench_mem.py n71582784 --mode bits      time the bit-ring step
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cupy as cp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from capstone_optimized.core import Graph, Polya_Process                # noqa: E402
from capstone_optimized.scale import quick_kernels as qk                # noqa: E402

from capstone_optimized.scale.paths import CACHE

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("--mode", default="bits", choices=["repo", "bits"])
ap.add_argument("--M", type=int, default=72)
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--time-last", type=int, default=5)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--verify", action="store_true")
args = ap.parse_args()

cdir = os.path.join(CACHE, args.cache)
indptr = cp.asarray(np.load(os.path.join(cdir, "indptr.npy")))
indices = cp.asarray(np.load(os.path.join(cdir, "indices.npy"), mmap_mode="r"))
n = indptr.size - 1


def run_repo(steps, M):
    g = Graph(num_nodes=n, num_colors=2, use_gpu=True, node_urns=cp.ones((n, 2), cp.int32))
    g.release_edge_buffer()
    g.csr_indptr, g.csr_indices = indptr, indices
    proc = Polya_Process(g, delta=cp.ones((n, 2), cp.int32), memory_enabled=True, memory_decay_time=M)
    cp.random.seed(args.seed)
    t = []
    for s in range(steps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        proc.step()
        cp.cuda.Device().synchronize()
        t.append(time.perf_counter() - t0)
    return g.node_urns, t


def run_bits(steps, M, use_array_rand):
    # the repo's step_count starts at 1: step k (1-based) uses ring slot k % M and expires once k > M
    words = cp.full(n, 1 | (1 << 16), cp.uint32)
    ring = cp.zeros((M, (n + 31) // 32), cp.uint32)
    draws = cp.empty(n, cp.uint8)
    cp.random.seed(args.seed)
    t = []
    for s in range(steps):
        k = s + 1
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        rnd = cp.random.rand(n) if use_array_rand else None
        qk.draw_u16(indptr, indices, words, 1, draws, rnd, seed=args.seed, step=s)
        qk.apply_u16_mem(words, draws, ring, k % M, k > M)
        cp.cuda.Device().synchronize()
        t.append(time.perf_counter() - t0)
    return qk.unpack_u16(words), t


if args.verify:
    ref, _ = run_repo(args.steps, args.M)
    got, _ = run_bits(args.steps, args.M, True)
    ok = bool(cp.array_equal(ref, got))
    print(json.dumps({"cache": args.cache, "n": n, "M": args.M, "steps": args.steps, "identical": ok,
                      "max_balls": int(ref.sum(axis=1).max().get())}))
    sys.exit(0 if ok else 1)

free0 = cp.cuda.runtime.memGetInfo()[0]
if args.mode == "repo":
    _, t = run_repo(args.steps, args.M)
else:
    _, t = run_bits(args.steps, args.M, False)
last = t[-args.time_last:]
print(json.dumps({"cache": args.cache, "n": n, "mode": args.mode, "M": args.M, "steps": args.steps,
                  "step_ms_last": round(1000 * float(np.median(last)), 1),
                  "step_ms_first": round(1000 * t[0], 1),
                  "total_s": round(sum(t), 2), "free_vram_gb_at_start": round(free0 / 2**30, 2)}), flush=True)
