"""
Checks breadth-first renumbering on the GPU (Graph.bfs_order / Graph.renumber) against scipy:
  - the order equals scipy.sparse.csgraph.breadth_first_order from the highest-degree node
  - the renumbered CSR equals scipy's A[p][:, p] with sorted rows
  - draws on the renumbered graph equal the original draws, permuted, given the same random number per node
and reports the draw kernel's time before and after (the renumbered graph has better cache locality).

  python src/capstone_optimized/experiments/test_bfs_renumber.py --batches 64 --sizes 4,64,1024,16384,262144
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from capstone_optimized.cupy_fallback import cp
from capstone_optimized.core import Graph, Barabasi_Albert_Growth
from capstone_optimized.core import csr_kernels

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=32)
ap.add_argument("--sizes", default="8,128,2048,32768")
args = ap.parse_args()
sync = cp.cuda.Device().synchronize


def init_urns(b_a, bs):
    b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, cp.ones((bs, 2), dtype=cp.int32)])
    b_a.graph.num_nodes += bs


cp.random.seed(4)
g = Graph(num_nodes=64, num_colors=2, use_gpu=True, node_urns=cp.ones((64, 2), dtype=cp.int32))
growth = Barabasi_Albert_Growth(g, m=3, initial_nodes=64)
for bs in [int(s) for s in args.sizes.split(",")]:
    for _ in range(args.batches):
        growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
growth.finalize_growth()
n = g.num_nodes
g.node_urns = cp.random.randint(0, 50, size=(n, 2)).astype(cp.int32)
rnd = cp.random.rand(n)

indptr_h, indices_h = cp.asnumpy(g.csr_indptr), cp.asnumpy(g.csr_indices)
A = sp.csr_matrix((np.ones(indices_h.size, np.int8), indices_h, indptr_h), shape=(n, n))
t = time.time()
ref = csgraph.breadth_first_order(A, int(np.argmax(np.diff(indptr_h))), directed=False, return_predecessors=False)
t_ref = time.time() - t


def kernel_ms(graph, rand_vals, reps=5):
    ts = []
    for _ in range(reps + 1):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        csr_kernels.draw_two_colour(graph.csr_indptr, graph.csr_indices, graph.node_urns, rand_vals)
        e1.record()
        e1.synchronize()
        ts.append(cp.cuda.get_elapsed_time(e0, e1))
    return sorted(ts[1:])[reps // 2]


draws0 = csr_kernels.draw_two_colour(g.csr_indptr, g.csr_indices, g.node_urns, rnd)
ms0 = kernel_ms(g, rnd)
sync()
t = time.time()
order = g.bfs_order()
sync()
t_bfs = time.time() - t
same_order = np.array_equal(cp.asnumpy(order), ref)
t = time.time()
g.renumber(order)
sync()
t_ren = time.time() - t
B = A[ref][:, ref]
B.sort_indices()
same_csr = np.array_equal(cp.asnumpy(g.csr_indptr), B.indptr) and np.array_equal(cp.asnumpy(g.csr_indices), B.indices)
rnd = rnd[order]
draws1 = csr_kernels.draw_two_colour(g.csr_indptr, g.csr_indices, g.node_urns, rnd)
same_draws = bool((draws1 == draws0[order]).all())
ms1 = kernel_ms(g, rnd)
print(f"n={n:,}: scipy BFS {t_ref:.1f}s, GPU BFS {t_bfs:.2f}s + renumber {t_ren:.2f}s")
print(f"order identical: {same_order}  CSR identical: {same_csr}  draws identical: {same_draws}")
print(f"draw kernel {ms0:.1f} ms -> {ms1:.1f} ms ({100 * (ms1 / ms0 - 1):+.0f}%)")
print("PASS" if same_order and same_csr and same_draws else "FAIL")
