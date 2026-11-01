"""Finite memory: the engine (enable_memory) against the repo's Polya_Process(memory_enabled=True) on the same
random numbers, urns compared after every step; then Philox with memory identical across placements."""
import argparse
import hashlib
import os

import cupy as cp
import numpy as np

from capstone_optimized.core import Graph, Polya_Process
from capstone_optimized.scale.engine import Engine

from capstone_optimized.scale.paths import CACHE, DATA
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="n17895744")
ap.add_argument("--M", type=int, default=3)
ap.add_argument("--steps", type=int, default=9)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
c = os.path.join(CACHE, args.name)
split = os.path.join(DATA, args.name + "_split")

indptr = cp.asarray(np.load(os.path.join(c, "indptr.npy")))
indices = cp.asarray(np.load(os.path.join(c, "indices.npy")))
n = indptr.size - 1
g = Graph(num_nodes=n, num_colors=2, use_gpu=True, node_urns=cp.ones((n, 2), cp.int32), initial_edge_capacity=1)
g.csr_indptr, g.csr_indices = indptr, indices
proc = Polya_Process(g, delta=cp.ones((n, 2), cp.int32), memory_enabled=True, memory_decay_time=args.M,
                     mem_decay=cp.ones((n, 2), cp.int32))
cp.random.seed(args.seed)
ref = []
for _ in range(args.steps):
    proc.step()
    ref.append(md5(cp.asnumpy(g.node_urns).astype(np.uint16)))
del proc, g, indptr, indices
cp.get_default_memory_pool().free_all_blocks()

eng = Engine(split, placement="vram", block_mb=8)
eng.enable_memory(args.M)
cp.random.seed(args.seed)
got = []
for _ in range(args.steps):
    eng.step(cp.random.rand(n))
    got.append(md5(eng.urns.get()))
print(f"memory M={args.M}: engine urns equal to Polya_Process after every step: {got == ref} "
      f"({sum(a == b for a, b in zip(got, ref))}/{args.steps})", flush=True)
eng.close()

hs = {}
for pl in ("vram", "mixed"):
    eng = Engine(split, placement=pl, block_mb=8, pinned_budget_gb=0.1, ram_budget_gb=0.1, seed=3)
    eng.enable_memory(args.M, ring_path=os.path.join(DATA, "ring_test.u8") if pl == "mixed" else None)
    eng.run(args.steps)
    hs[pl] = md5(eng.urns.get())
    eng.close()
os.remove(os.path.join(DATA, "ring_test.u8"))
print("philox + memory, vram vs mixed (ring in a file):", hs, "same:", len(set(hs.values())) == 1)
