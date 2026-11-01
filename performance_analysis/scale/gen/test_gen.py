"""
Checks that scale.gen produces exactly the repo's graph.

  --case cache   seed 493, batches 2 and 32, against data/n1118544_split and data/n17895744_split (converted from
                 the CSRs the repo's growth cached on 2026-10-29)
  --case repo    seed and batches given, against the repo's growth run in this process (worktree code:
                 Barabasi_Albert_Growth.grow_batch + finalize_growth), converted with split_format.from_csr
Two split graphs are equal when meta sizes, in_ptr and in_idx are equal and every out row holds the same targets
(the generator keeps a node's targets in the order it chose them, the converter in column order).
Run under the GPU lock from the worktree root with PYTHONPATH=src.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import cupy as cp

from capstone_optimized.scale import gen, split_format as sf

SCRATCH = r"C:\Users\dougl\AppData\Local\Temp\claude\c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks\7bf0ea5c-bb18-4a6c-8e60-f701292fffbe\scratchpad"
DATA = os.path.join(SCRATCH, "data")


def compare_split(d1, d2, chunk_rows=1 << 23):
    m1, o1, p1, i1 = sf.open_arrays(d1)
    m2, o2, p2, i2 = sf.open_arrays(d2)
    errs = []
    for k in ("n", "m", "init", "n_out", "nnz_in"):
        if m1[k] != m2[k]:
            errs.append(f"meta {k}: {m1[k]} != {m2[k]}")
    if errs:
        return errs
    n, m, init = m1["n"], m1["m"], m1["init"]
    for a in range(0, n + 1, chunk_rows):
        b = min(n + 1, a + chunk_rows)
        if not np.array_equal(p1[a:b], p2[a:b]):
            errs.append(f"in_ptr differs in rows {a}..{b}")
            break
    for a in range(0, m1["nnz_in"], chunk_rows * 4):
        b = min(m1["nnz_in"], a + chunk_rows * 4)
        if not np.array_equal(i1[a:b], i2[a:b]):
            errs.append(f"in_idx differs in entries {a}..{b}")
            break
    rows = n - init
    for a in range(0, rows, chunk_rows):
        b = min(rows, a + chunk_rows)
        x = np.sort(np.asarray(o1[a * m:b * m]).reshape(-1, m), axis=1)
        y = np.sort(np.asarray(o2[a * m:b * m]).reshape(-1, m), axis=1)
        if not np.array_equal(x, y):
            bad = np.flatnonzero((x != y).any(axis=1))
            errs.append(f"out rows differ, first at node {init + a + bad[0]}: {x[bad[0]]} vs {y[bad[0]]}")
            break
    return errs


def repo_split(seed, batches, out_dir):
    """Grows the graph with the repo's code (build_graph.py harness) and converts its CSR to the split format."""
    from capstone_optimized.core import Graph, Barabasi_Albert_Growth

    def init_urns(b_a, bs):
        added = cp.ones((bs, b_a.graph.num_colors), dtype=cp.int32)
        b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, added])
        b_a.graph.num_nodes = b_a.graph.num_nodes + bs

    cp.random.seed(seed)
    g = Graph(num_nodes=64, num_colors=2, use_gpu=True, node_urns=cp.ones((64, 2), dtype=cp.int32))
    growth = Barabasi_Albert_Growth(g, m=3, initial_nodes=64)
    t = time.time()
    for bs in gen.SIZES:
        for _ in range(batches):
            growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
    growth.finalize_growth()
    cp.cuda.Device().synchronize()
    t_repo = time.time() - t
    if g.csr_indices is not None:
        indptr, indices, data = cp.asnumpy(g.csr_indptr), g.csr_indices, None
        indices = cp.asnumpy(indices) if isinstance(indices, cp.ndarray) else np.asarray(indices)
    else:
        A = g.adj_matrix
        indptr, indices, data = cp.asnumpy(A.indptr), cp.asnumpy(A.indices), cp.asnumpy(A.data)
    meta = sf.from_csr(out_dir, indptr, indices, m=3, init=64, data=data, source=f"repo growth seed {seed}")
    weighted = 0 if data is None else int((data != 1).sum())
    del g, growth
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return meta, t_repo, weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["cache", "repo"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--shift", type=int, default=None, help="pinned chunk size 2^shift entries (test boundaries)")
    ap.add_argument("--key-cap", type=int, default=None, help="max in-part entries per fill pass")
    ap.add_argument("--dev-cache-gb", type=float, default=3.0, help="0 sends every lookup to pinned memory")
    args = ap.parse_args()
    if args.shift:
        gen.PinnedChunks.SHIFT = args.shift
    kw = dict(dev_cache_gb=args.dev_cache_gb, key_cap=args.key_cap)
    ok = True
    if args.case == "cache":
        for batches, ref in [(2, "n1118544_split"), (32, "n17895744_split")]:
            d = os.path.join(DATA, f"gen_test_b{batches}")
            meta = gen.generate(d, batches, seed=493, **kw)
            errs = compare_split(d, os.path.join(DATA, ref))
            errs += sf.validate(d)
            print(f"seed 493 batches {batches}: {'IDENTICAL' if not errs else errs[:5]}  gen {meta['timings']}")
            ok &= not errs
            if not args.keep:
                shutil.rmtree(d)
    else:
        dr = os.path.join(DATA, f"repo_test_s{args.seed}_b{args.batches}")
        dg = os.path.join(DATA, f"gen_test_s{args.seed}_b{args.batches}")
        rmeta, t_repo, weighted = repo_split(args.seed, args.batches, dr)
        meta = gen.generate(dg, args.batches, seed=args.seed, **kw)
        errs = compare_split(dg, dr) + sf.validate(dg)
        print(f"seed {args.seed} batches {args.batches} (n={meta['n']:,}, repo duplicate-edge entries {weighted}): "
              f"{'IDENTICAL' if not errs else errs[:5]}  repo growth+csr {t_repo:.1f} s, gen {meta['timings']}")
        ok &= not errs
        if not args.keep:
            shutil.rmtree(dr)
            shutil.rmtree(dg)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
