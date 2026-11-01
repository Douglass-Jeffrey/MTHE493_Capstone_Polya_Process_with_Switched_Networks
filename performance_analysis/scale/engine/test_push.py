"""Push rows (Engine(push_k=K)): urns must equal the repo's step on the same random numbers, and equal the pull-only
engine with Philox, for VRAM and streamed placements, with and without finite memory."""
import argparse
import hashlib
import os

import cupy as cp
import numpy as np

from capstone_optimized.core import csr_kernels
from capstone_optimized.scale.engine import Engine

S = r"C:\Users\dougl\AppData\Local\Temp\claude\c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks\7bf0ea5c-bb18-4a6c-8e60-f701292fffbe\scratchpad"
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="n17895744")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--push-k", type=int, default=2 << 20)
ap.add_argument("--seed", type=int, default=5)
args = ap.parse_args()
md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
split = os.path.join(S, "data", args.name + "_split")
c = os.path.join(S, "opt", "cache", args.name)

indptr = cp.asarray(np.load(os.path.join(c, "indptr.npy")))
indices = cp.asarray(np.load(os.path.join(c, "indices.npy")))
n = indptr.size - 1
urns = cp.ones((n, 2), cp.int32)
delta = cp.ones((n, 2), cp.int32)
cp.random.seed(args.seed)
ref = []
for _ in range(args.steps):
    draws = csr_kernels.draw_two_colour(indptr, indices, urns, cp.random.rand(n))
    csr_kernels.apply_draws(draws, delta, urns)
    ref.append(md5(cp.asnumpy(urns).astype(np.uint16)))
del indptr, indices, urns, delta, draws
cp.get_default_memory_pool().free_all_blocks()

eng = Engine(split, placement="vram", block_mb=8, push_k=args.push_k)
cp.random.seed(args.seed)
got = []
for _ in range(args.steps):
    eng.step(cp.random.rand(n))
    got.append(md5(eng.urns.get()))
print(f"push K={eng.K:,}: urns equal to the repo after every step: {got == ref}", flush=True)
eng.close()

for mem in (0, 3):
    hs = {}
    for name, kw in (("pull vram", dict(placement="vram")), ("push vram", dict(placement="vram", push_k=args.push_k)),
                     ("push stream", dict(placement="stream", push_k=args.push_k)),
                     ("push auto mixed", dict(placement="mixed", pinned_budget_gb=0.1, ram_budget_gb=0.1, push_k=-1))):
        eng = Engine(split, block_mb=8, seed=9, **kw)
        if mem:
            eng.enable_memory(mem)
        eng.run(args.steps + 3)
        hs[f"{name} (K={eng.K:,})"] = md5(eng.urns.get())
        eng.close()
        cp.get_default_memory_pool().free_all_blocks()
    print(f"memory {mem}: philox same for all: {len(set(hs.values())) == 1} {hs}", flush=True)
