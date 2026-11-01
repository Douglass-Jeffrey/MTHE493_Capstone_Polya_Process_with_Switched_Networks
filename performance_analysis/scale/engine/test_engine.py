"""
Checks the out-of-core engine against the repo's fused step, and its placements against each other.

(a) repo: cached CSR in VRAM, int32 urns, csr_kernels.draw_two_colour + apply_draws, rand = cp.random.rand(n)
    after cp.random.seed(seed). engine: same rand arrays. Urns must be equal after every step checked.
(b) Philox: the engine's urns after K steps must be identical for every placement.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cupy as cp
import numpy as np

from capstone_optimized.core import csr_kernels
from capstone_optimized.scale.engine import Engine

S = r"C:\Users\dougl\AppData\Local\Temp\claude\c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks\7bf0ea5c-bb18-4a6c-8e60-f701292fffbe\scratchpad"
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="n17895744")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--block-mb", type=float, default=None)
ap.add_argument("--block-rows", type=int, default=1 << 20)
ap.add_argument("--placements", default="vram,pinned,stream,mixed")
ap.add_argument("--skip-repo", action="store_true")
ap.add_argument("--cpu-blocks", type=int, default=3)
args = ap.parse_args()
split = os.path.join(S, "data", args.name + "_split")
out = {"name": args.name, "steps": args.steps}


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# (a) repo reference
if not args.skip_repo:
    c = os.path.join(S, "opt", "cache", args.name)
    indptr = cp.asarray(np.load(os.path.join(c, "indptr.npy")))
    indices = cp.asarray(np.load(os.path.join(c, "indices.npy")))
    n = indptr.size - 1
    urns = cp.ones((n, 2), cp.int32)
    delta = cp.ones((n, 2), cp.int32)
    cp.random.seed(args.seed)
    ref = []
    for s in range(args.steps):
        rand = cp.random.rand(n)
        draws = csr_kernels.draw_two_colour(indptr, indices, urns, rand)
        csr_kernels.apply_draws(draws, delta, urns)
        ref.append(md5(cp.asnumpy(urns).astype(np.uint16)))
    ref_final = cp.asnumpy(urns)
    del indptr, indices, urns, delta, rand, draws
    cp.get_default_memory_pool().free_all_blocks()

    for pl in args.placements.split(","):
        eng = Engine(split, placement=pl, block_rows=args.block_rows, block_mb=args.block_mb, pinned_budget_gb=0.1 if pl == "mixed" else 8, ram_budget_gb=0.1)
        cp.random.seed(args.seed)
        got = []
        for s in range(args.steps):
            rand = cp.random.rand(eng.n)
            eng.step(rand)
            got.append(md5(eng.urns.get()))
        same = got == ref
        out[f"ext_{pl}"] = same
        print(f"(a) {pl}: urns equal to repo after every step: {same}", flush=True)
        eng.close()
        del eng
        cp.get_default_memory_pool().free_all_blocks()

# (b) Philox across placements
hashes = {}
for pl in args.placements.split(","):
    eng = Engine(split, placement=pl, block_rows=args.block_rows, block_mb=args.block_mb, pinned_budget_gb=0.1 if pl == "mixed" else 8, ram_budget_gb=0.1,
                 seed=args.seed)
    per, wall = eng.run(args.steps)
    hashes[pl] = md5(eng.urns.get())
    red = float(eng.urns[:, 0].astype(cp.float64).sum() / eng.urns.astype(cp.float64).sum())
    print(f"(b) {pl}: md5 {hashes[pl]} red share {red:.4f} step times {['%.3f' % x for x in per]} "
          f"overflow {eng.overflowed()}", flush=True)
    eng.close()
    del eng
    cp.get_default_memory_pool().free_all_blocks()
# (c) CPU+GPU: the last blocks drawn on the CPU
if args.cpu_blocks:
    eng = Engine(split, placement="vram", block_rows=args.block_rows, block_mb=args.block_mb, seed=args.seed)
    eng.enable_cpu(args.cpu_blocks)
    per, wall = eng.run(args.steps)
    hashes["vram+cpu"] = md5(eng.urns.get())
    print(f"(c) vram+cpu ({args.cpu_blocks} blocks on the CPU): md5 {hashes['vram+cpu']} step times "
          f"{['%.3f' % x for x in per]} cpu (apply, draw) {[('%.3f' % a, '%.3f' % b) for a, b in eng.cpu_times]}",
          flush=True)
    eng.close()
    del eng
    cp.get_default_memory_pool().free_all_blocks()
out["philox_same_all"] = len(set(hashes.values())) == 1
out["philox_md5"] = hashes
print(json.dumps(out))
