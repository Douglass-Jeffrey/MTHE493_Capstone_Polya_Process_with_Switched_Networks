"""Relabel the nodes of a cached CSR for locality and save it as cache/<n>_<perm>/.

p[new] = old. Orders: random (worst case), deg (degree descending), bfs (breadth-first from the top hub),
rcm (reverse Cuthill-McKee), degbfs (BFS that visits neighbours in degree order: hubs early in each level).
--sort-rows also sorts every row's neighbour list after relabelling.
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("perm", choices=["random", "deg", "bfs", "rcm", "orig"])
ap.add_argument("--sort-rows", action="store_true")
args = ap.parse_args()
D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
src = os.path.join(WORK, "cache", args.cache)
indptr = np.load(os.path.join(src, "indptr.npy"))
indices = np.load(os.path.join(src, "indices.npy"))
n, nnz = indptr.size - 1, int(indptr[-1])
deg = np.diff(indptr)

t0 = time.time()
if args.perm == "orig":
    p = np.arange(n)
elif args.perm == "random":
    p = np.random.default_rng(0).permutation(n)
elif args.perm == "deg":
    p = np.argsort(-deg, kind="stable")
else:
    A = sp.csr_matrix((np.ones(nnz, np.int8), indices, indptr), shape=(n, n))
    if args.perm == "bfs":
        p = csgraph.breadth_first_order(A, int(np.argmax(deg)), directed=False, return_predecessors=False)
    else:
        p = csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    if p.size < n:                                         # unreached nodes (none for a connected BA graph)
        seen = np.zeros(n, bool); seen[p] = True
        p = np.concatenate([p, np.flatnonzero(~seen)])
    del A
t_perm = time.time() - t0

inv = np.empty(n, np.int32)
inv[p] = np.arange(n, dtype=np.int32)
new_deg = deg[p]
new_indptr = np.zeros(n + 1, np.int64)
np.cumsum(new_deg, out=new_indptr[1:])
pos = np.repeat(indptr[p].astype(np.int64) - new_indptr[:-1], new_deg)
pos += np.arange(nnz, dtype=np.int64)
new_indices = inv[indices[pos]]
del pos
if args.sort_rows:                                         # sort neighbours inside each row
    rowid = np.repeat(np.arange(n, dtype=np.int64), new_deg)
    key = rowid * n + new_indices
    del rowid
    key.sort()
    new_indices = (key % n).astype(np.int32)
    del key
t_total = time.time() - t0

name = f"{args.cache}_{args.perm}" + ("_s" if args.sort_rows else "")
out = os.path.join(WORK, "cache", name)
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, "indptr.npy"), new_indptr.astype(np.int32))
np.save(os.path.join(out, "indices.npy"), new_indices)
np.save(os.path.join(out, "perm.npy"), p.astype(np.int32))
# locality summary: median |i - j| over all edges, and share of edges whose endpoints share a 64 B urn block
r = np.repeat(np.arange(n, dtype=np.int64), new_deg)
gap = np.abs(r - new_indices)
meta = {"perm": args.perm, "sort_rows": args.sort_rows, "perm_s": t_perm, "total_s": t_total,
        "median_gap": float(np.median(gap)), "p90_gap": float(np.percentile(gap, 90)),
        "same_64B_block": float(np.mean((r >> 3) == (new_indices >> 3)))}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print(name, json.dumps(meta))
