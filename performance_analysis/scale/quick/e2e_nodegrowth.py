"""
Node-growth intervention in the middle of a run with finite memory, as experiments/test_node_growth_intervention_polya.py
does it, with last night's code (--src orig) or this branch (--src new). The intervention extends
polya.memory_arr with hstack, which on this branch goes through the bit ring's pack/unpack.
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", choices=["orig", "new"], required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BASE = os.path.join(os.environ.get("POLYA_WORK") or os.path.join(REPO, "work"), "base_src", "src")   # as scale/paths.py, computed here because the package is imported after choosing its source
sys.path.insert(0, BASE if args.src == "orig" else os.path.join(REPO, "src"))
from capstone_optimized.cupy_fallback import cp                                   # noqa: E402
from capstone_optimized.core import (Graph, Barabasi_Albert_Growth, Polya_Process,  # noqa: E402
                                     Switch_Network_Node_Growth_Intervention)

cp.random.seed(3)
g = Graph(num_nodes=64, num_colors=2, use_gpu=True, node_urns=cp.ones((64, 2), dtype=cp.int32))
growth = Barabasi_Albert_Growth(g, m=3, initial_nodes=64)
for bs in (8, 128, 2048):
    for _ in range(4):
        growth.grow_batch(batch_size=bs, urn_addition_callback=lambda b, k: (
            setattr(b.graph, "node_urns", cp.vstack([b.graph.node_urns, cp.ones((k, 2), cp.int32)])),
            setattr(b.graph, "num_nodes", b.graph.num_nodes + k)))
iv = Switch_Network_Node_Growth_Intervention(g)          # needs the edge buffer, so before finalize_growth
try:
    growth.finalize_growth(release_edges=False)   # keeps the edge buffer for the mid-run interventions
except TypeError:   # code before the fix, where finalize_growth always frees it: build the CSR and keep the edges
    g.build_csr()
n = g.num_nodes
p = Polya_Process(g, memory_enabled=True, memory_decay_time=6,
                  mem_decay=cp.random.randint(0, 2, size=(n, 2)).astype(cp.int32))
status = "ok"
try:
    for s in range(30):
        p.step()
        if s in (8, 17):
            if g.edges is None:
                status = "edge buffer released by finalize_growth: mid-run intervention cannot add edges"
                break
            iv.random_intervention_step(num_connections=3, intervener_urn=[5, 1], mid_polya_intervention=True,
                                        polya_process=p, num_interventions=4)
except Exception as e:
    status = f"{type(e).__name__}: {e}"
np.save(args.out + "_urns.npy", cp.asnumpy(g.node_urns))
print(args.src, "nodes", g.num_nodes, "status:", status, "balls", int(g.node_urns.sum()), "min count", int(g.node_urns.min()))
