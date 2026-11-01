"""
Workstream A: time and verify step variants on a cached CSR held in VRAM.

  python bench_quick.py n71582784 --variants repo,i32_arr,u16_arr --gran 64 --steps 10
  python bench_quick.py n1118544 --verify

Variants: repo (csr_kernels.draw_two_colour + apply_draws), i32_arr / i32_phx (int2 urns, random array or
Philox), u16_arr / u16_phx (packed 16-bit urns), u16_R{2,4,8}_phx (that many runs sharing one gather).
Prints one JSON line per variant: median step, draw kernel and apply times, per-run step time.
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
from capstone_optimized.core import csr_kernels                       # noqa: E402
from capstone_optimized.scale import quick_kernels as qk              # noqa: E402

CACHE = r"C:\Users\dougl\AppData\Local\Temp\claude\c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks\7bf0ea5c-bb18-4a6c-8e60-f701292fffbe\scratchpad\opt\cache"

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("--variants", default="repo,i32_arr,i32_phx,u16_arr,u16_phx,u16_R2_phx,u16_R4_phx,u16_R8_phx")
ap.add_argument("--gran", type=int, default=64, help="cudaLimitMaxL2FetchGranularity (bytes)")
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--verify", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--lanes", type=int, default=4)
args = ap.parse_args()

gran = qk.set_l2_fetch_granularity(args.gran)
cdir = os.path.join(CACHE, args.cache)
indptr = cp.asarray(np.load(os.path.join(cdir, "indptr.npy")))
indices = cp.asarray(np.load(os.path.join(cdir, "indices.npy"), mmap_mode="r"))
assert indptr.dtype == cp.int32 and indices.dtype == cp.int32
n = indptr.size - 1


def run(variant, steps, record=True):
    """Runs `steps` steps from fresh urns; returns (final urns as (n, R, 2) int32, timings)."""
    R = 1
    if variant.startswith("u16_R"):
        R = int(variant.split("_")[1][1:])
    # buffers cached from the previous variant would otherwise push this one's arrays into system RAM
    cp.get_default_memory_pool().free_all_blocks()
    cp.random.seed(args.seed)
    ev = [cp.cuda.Event() for _ in range(4)]
    times = []
    if variant in ("repo", "i32_arr", "i32_phx"):
        urns = cp.ones((n, 2), cp.int32)
    else:
        words = cp.full(n * R, 1 | (1 << 16), cp.uint32)
    draws = cp.empty(n * R, cp.uint8)
    delta = cp.ones((n, 2), cp.int32) if variant == "repo" else None
    for s in range(steps):
        ev[0].record()
        rnd = cp.random.rand(n * R) if variant in ("repo", "i32_arr", "u16_arr") else None
        ev[1].record()
        if variant == "repo":
            d32 = csr_kernels.draw_two_colour(indptr, indices, urns, rnd)
            ev[2].record()
            csr_kernels.apply_draws(d32, delta, urns)
        elif variant.startswith("i32"):
            qk.draw_i32(indptr, indices, urns, draws, rnd, seed=args.seed, step=s)
            ev[2].record()
            qk.apply_i32(urns, draws)
        else:
            qk.draw_u16(indptr, indices, words, R, draws, rnd, seed=args.seed, step=s, lanes=args.lanes)
            ev[2].record()
            qk.apply_u16(words, draws)
        ev[3].record()
        ev[3].synchronize()
        times.append([cp.cuda.get_elapsed_time(ev[i], ev[i + 1]) for i in range(3)])
    if variant in ("repo", "i32_arr", "i32_phx"):
        final = urns.reshape(n, 1, 2)
    else:
        final = qk.unpack_u16(words).reshape(n, R, 2)
    return final, np.array(times)


if args.verify:
    k = args.steps
    ref, _ = run("repo", k)
    out = {}
    for v in ("i32_arr", "u16_arr"):
        u, _ = run(v, k)
        out[v + "==repo"] = bool(cp.array_equal(u, ref))
    p1, _ = run("i32_phx", k)
    p2, _ = run("u16_phx", k)
    out["u16_phx==i32_phx"] = bool(cp.array_equal(p1, p2))
    for R in (2, 4, 8, 16):
        pr, _ = run(f"u16_R{R}_phx", k)
        out[f"R{R} run0==R1"] = bool(cp.array_equal(pr[:, :1], p2))
        out[f"R{R} runs differ"] = bool(not cp.array_equal(pr[:, 0], pr[:, R - 1]))
    out["red share after %d steps (repo)" % k] = float((ref[..., 0].sum() / ref.sum()).get())
    print(json.dumps({"cache": args.cache, "n": n, "verify": out}))
    sys.exit(0 if all(v for kk, v in out.items() if isinstance(v, bool)) else 1)

for v in args.variants.split(","):
    run(v, 1)                                            # compile + warm the caches
    _, t = run(v, args.warmup + args.steps)
    t = t[args.warmup:]
    R = int(v.split("_")[1][1:]) if v.startswith("u16_R") else 1
    step = float(np.median(t.sum(axis=1)))
    print(json.dumps({"cache": args.cache, "n": n, "variant": v, "R": R, "gran": gran, "tag": args.tag, "lanes": args.lanes,
                      "step_ms": round(step, 2), "rand_ms": round(float(np.median(t[:, 0])), 2),
                      "draw_ms": round(float(np.median(t[:, 1])), 2), "apply_ms": round(float(np.median(t[:, 2])), 2),
                      "per_run_ms": round(step / R, 2), "draw_min_ms": round(float(t[:, 1].min()), 2)}), flush=True)
