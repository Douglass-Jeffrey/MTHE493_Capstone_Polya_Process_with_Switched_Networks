"""CPU (NumPy/SciPy) path of the same benchmark: hide the GPU so cupy_fallback drops to NumPy."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, json, sys, time

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
ORIG_SRC = os.path.join(WORK, "orig_src")   # the code at fd79e4d, which these measurements were taken on
sys.path.insert(0, ORIG_SRC)
from capstone_optimized import cupy_fallback as cf
assert not cf.CUPY_AVAILABLE, "expected the NumPy fallback"
np = cf.cp
from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=64)
ap.add_argument("--sizes", default="8,128,2048,32768,524288")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--out")
args = ap.parse_args()


def init_urns(b_a, bs):
    added = np.ones((bs, b_a.graph.num_colors), dtype=np.int32)
    b_a.graph.node_urns = np.vstack([b_a.graph.node_urns, added])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs


graph = Graph(num_nodes=64, num_colors=2, use_gpu=True, node_urns=np.ones((64, 2), dtype=np.int32))
growth = Barabasi_Albert_Growth(graph, m=3, initial_nodes=64)
t0 = time.time()
for bs in [int(s) for s in args.sizes.split(",")]:
    for _ in range(args.batches):
        growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
t1 = time.time()
growth.finalize_growth()
t2 = time.time()
polya = Polya_Process(graph, delta=np.ones((graph.num_nodes, 2), dtype=np.int32))
steps = []
for _ in range(args.steps):
    ts = time.time()
    polya.step()
    steps.append(time.time() - ts)
log = {"nodes": int(graph.num_nodes), "edges": int(graph.num_edges), "growth_s": t1 - t0, "csr_s": t2 - t1,
       "steps": steps}
print(json.dumps(log))
json.dump(log, open(args.out, "w"))
