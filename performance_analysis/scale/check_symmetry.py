"""
Independent check of a split-format graph: every edge appears once from each end. For sampled rows v,
each out target t of v must list v in t's in row, and each in neighbour u >= init of v must list v in u's out
row. Also sizes, in_ptr monotone at sampled points, and the clique rows. Reads by memory map; no GPU.

  python check_symmetry.py <data dir> [samples]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from capstone_optimized.scale import split_format as sf        # noqa: E402

d = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
t0 = time.time()
meta, out, in_ptr, in_idx = sf.open_arrays(d)
n, m, init = meta["n"], meta["m"], meta["init"]
errs = []
if out.size != (n - init) * m or in_ptr[-1] != in_idx.size or in_idx.size - out.size != init * (init - 1):
    errs.append("sizes")
rng = np.random.default_rng(1)
rows = np.unique(np.concatenate([np.arange(init + 8), rng.integers(init, n, k),
                                 np.geomspace(init, n - 1, 200).astype(np.int64), [n - 1]]))


def out_row(v):
    return np.asarray(out[(v - init) * m:(v - init + 1) * m]) if v >= init else np.zeros(0, np.int64)


def in_row(v):
    return np.asarray(in_idx[in_ptr[v]:in_ptr[v + 1]])


checked = 0
for v in rows:
    v = int(v)
    o, i = out_row(v), in_row(v)
    if i.size and (np.diff(i) < 0).any():
        errs.append(f"row {v}: in row not sorted")
    for t in o:
        r = in_row(int(t))
        j = np.searchsorted(r, v)
        if not (j < r.size and r[j] == v):
            errs.append(f"edge {v}->{t}: {v} missing from in row of {t}")
        checked += 1
    for u in i:
        u = int(u)
        if u >= init:
            if v not in out_row(u):
                errs.append(f"row {v}: in neighbour {u} did not choose {v}")
        elif not (v < init):
            errs.append(f"row {v}: clique node {u} in the in row of a non-clique node")
        checked += 1
    if v < init and np.setdiff1d(np.setdiff1d(np.arange(init), [v]), i).size:
        errs.append(f"clique row {v}: missing clique neighbours")
    if len(errs) > 20:
        break
print(f"{d}: n={n:,} rows checked {rows.size}, edge ends checked {checked:,}, {time.time() - t0:.0f} s, "
      + ("OK" if not errs else f"{len(errs)} problems: {errs[:5]}"))
sys.exit(1 if errs else 0)
