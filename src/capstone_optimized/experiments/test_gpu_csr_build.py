"""
Checks the GPU CSR build (Graph.build_csr -> csr_kernels.csr_from_edges) against scipy's coo -> csr on the
same edge list: indptr and indices must be identical, with the indices in VRAM and in pinned host memory.
Also checks that a graph with duplicate edges falls back to scipy, which sums them into weights > 1.

  python src/capstone_optimized/experiments/test_gpu_csr_build.py --batches 64 --sizes 4,64,1024,16384,262144
"""
import argparse, os, sys, time
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from capstone_optimized.cupy_fallback import cp
from capstone_optimized.core import Graph, Barabasi_Albert_Growth

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=32)
ap.add_argument("--sizes", default="8,128,2048,32768")
args = ap.parse_args()


def init_urns(b_a, bs):
    b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, cp.ones((bs, 2), dtype=cp.int32)])
    b_a.graph.num_nodes += bs


ok = True
for placement in ("device", "host"):
    cp.random.seed(3)
    g = Graph(num_nodes=64, num_colors=2, use_gpu=True, node_urns=cp.ones((64, 2), dtype=cp.int32))
    g.csr_placement = placement
    growth = Barabasi_Albert_Growth(g, m=3, initial_nodes=64)
    for bs in [int(s) for s in args.sizes.split(",")]:
        for _ in range(args.batches):
            growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
    n, m = g.num_nodes, g.num_edges
    e = cp.asnumpy(g.edges[:m])
    t = time.time()
    ref = csr_matrix((np.ones(m, np.float32), (e[:, 0], e[:, 1])), shape=(n, n))
    t_ref = time.time() - t
    del e
    cp.cuda.Device().synchronize()
    t = time.time()
    growth.finalize_growth()
    cp.cuda.Device().synchronize()
    t_gpu = time.time() - t
    ix = g.csr_indices if isinstance(g.csr_indices, np.ndarray) else cp.asnumpy(g.csr_indices)
    same = np.array_equal(cp.asnumpy(g.csr_indptr), ref.indptr) and np.array_equal(ix, ref.indices)
    ok &= same
    print(f"{placement}: n={n:,} nnz={m:,}  scipy {t_ref:.1f}s  GPU {t_gpu:.2f}s  identical: {same}")

# duplicate edges must take the scipy path, which sums them into weight > 1 (and the SpMM step)
h = Graph(num_nodes=1000, num_colors=2, use_gpu=True, initial_edge_capacity=10)
rng = np.random.default_rng(0)
i = rng.integers(0, 1000, 1_200_000).astype(np.int32)
j = rng.integers(0, 1000, 1_200_000).astype(np.int32)
h.add_edges(cp.asarray(i), cp.asarray(j))
h.build_csr()
ref = csr_matrix((np.ones(i.size, np.float32), (i, j)), shape=(1000, 1000))
A = h.adj_matrix
same = (h.csr_indices is None and np.array_equal(cp.asnumpy(A.indptr), ref.indptr)
        and np.array_equal(cp.asnumpy(A.indices), ref.indices) and np.array_equal(cp.asnumpy(A.data), ref.data))
ok &= same
print(f"duplicate edges fall back to scipy (max weight {float(A.data.max()):.0f}): {same}")
print("PASS" if ok else "FAIL")
