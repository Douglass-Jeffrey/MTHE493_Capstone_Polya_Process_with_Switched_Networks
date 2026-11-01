"""Bit-exactness check: the repo's Polya_Process.step (cupyx SpMM) vs the fused kernels, same seed, K steps.
Also checks gather() against A @ U + U for 2 and 3 colours with random urns."""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
ORIG_SRC = os.path.join(WORK, "orig_src")   # the code at fd79e4d, which these measurements were taken on
sys.path.insert(0, ORIG_SRC)   # the repo's step as it was (cupyx SpMM)
from capstone_optimized.cupy_fallback import cp
from capstone_optimized.core import Graph, Polya_Process
from cupyx.scipy.sparse import csr_matrix
import kern

D = os.path.dirname(os.path.abspath(__file__))
n_dir = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
c_dir = os.path.join(WORK, "cache", n_dir)
indptr = cp.asarray(np.load(os.path.join(c_dir, "indptr.npy")))
indices = cp.asarray(np.load(os.path.join(c_dir, "indices.npy")))
n = indptr.size - 1
A = csr_matrix((cp.ones(indices.size, cp.float32), indices, indptr), shape=(n, n))
ok = True

# 1. gather() == A @ U + U, random urns, 2 and 3 colours, every lane width
rs = cp.random.RandomState(7)
for c in (2, 3):
    U = rs.randint(0, 1000, size=(n, c)).astype(cp.int32)
    ref = A @ U.astype(cp.float64) + U
    for lanes in kern.LANE_OPTS:
        same = bool(cp.array_equal(kern.gather(indptr, indices, U, lanes=lanes), ref))
        ok &= same
        if not same:
            print(f"gather MISMATCH c={c} lanes={lanes}")
print(f"gather == A@U+U for c=2,3 and all lane widths: {ok}")

# 2. K Polya steps: repo step vs fused draw2 + apply_draws, identical seeds, non-uniform delta
delta = (rs.randint(1, 4, size=(n, 2))).astype(cp.int32)
start = rs.randint(1, 50, size=(n, 2)).astype(cp.int32)
g = Graph(num_nodes=n, num_colors=2, use_gpu=True, node_urns=start.copy())
g.adj_matrix = A
p = Polya_Process(g, delta=delta.copy())
cp.random.seed(1234)
for _ in range(K):
    p.step()
ref_urns = g.node_urns.copy()
for lanes in kern.LANE_OPTS:
    urns = start.copy()
    cp.random.seed(1234)
    for _ in range(K):
        rnd = cp.random.rand(n)
        draws = cp.empty(n, cp.int32)
        kern.draw2(indptr, indices, urns, rnd, draws, lanes=lanes)
        kern.apply_draws(draws, delta, np.int64(2), urns)
    same = bool(cp.array_equal(urns, ref_urns))
    ok &= same
    print(f"{K} steps, lanes={lanes:>2}: urns identical to repo step: {same}  "
          f"(balls added: {int((ref_urns - start).sum())})")
print("ALL OK" if ok else "FAILED")
