"""Grow a BA graph with the unmodified repo code (same settings as test_benchmark) and cache its CSR to .npy,
so every experiment runs on the exact matrix the benchmark multiplies."""
import argparse, json, os, sys, time
import numpy as np

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
ORIG_SRC = os.path.join(WORK, "orig_src")   # the code at fd79e4d, which these measurements were taken on
sys.path.insert(0, ORIG_SRC)
from capstone_optimized.cupy_fallback import cp
from capstone_optimized.core import Graph, Barabasi_Albert_Growth

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, required=True)
ap.add_argument("--seed", type=int, default=493)
args = ap.parse_args()

SIZES = [8, 128, 2048, 32768, 524288]
initial_nodes, num_colors = 64, 2
cp.random.seed(args.seed)


def init_urns(b_a, bs):
    added = cp.ones((bs, b_a.graph.num_colors), dtype=cp.int32)
    b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, added])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs


graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True,
              node_urns=cp.ones((initial_nodes, num_colors), dtype=cp.int32))
growth = Barabasi_Albert_Growth(graph, m=3, initial_nodes=initial_nodes)
t0 = time.time()
for bs in SIZES:
    for _ in range(args.batches):
        growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
cp.cuda.Device().synchronize()
t1 = time.time()
growth.finalize_growth()
cp.cuda.Device().synchronize()
t2 = time.time()

A = graph.adj_matrix
n = graph.num_nodes
out = os.path.join(WORK, "cache", f"n{n}")
os.makedirs(out, exist_ok=True)
data = cp.asnumpy(A.data)
np.save(os.path.join(out, "indptr.npy"), cp.asnumpy(A.indptr))
np.save(os.path.join(out, "indices.npy"), cp.asnumpy(A.indices))
vals, counts = np.unique(data, return_counts=True)
if len(vals) > 1:
    np.save(os.path.join(out, "data.npy"), data)
deg = np.diff(cp.asnumpy(A.indptr))
meta = {"n": n, "nnz": int(A.nnz), "batches": args.batches, "seed": args.seed,
        "growth_s": t1 - t0, "csr_s": t2 - t1, "weights": {str(v): int(c) for v, c in zip(vals, counts)},
        "max_degree": int(deg.max()), "mean_degree": float(deg.mean()),
        "indptr_dtype": str(A.indptr.dtype), "indices_dtype": str(A.indices.dtype)}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print(json.dumps(meta))
