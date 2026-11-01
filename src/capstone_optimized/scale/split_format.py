"""
"Split" adjacency format for graphs too large for VRAM (or RAM), written by the out-of-core generator and read
by the out-of-core step engine.

A Barabasi-Albert graph grown in batches has a fixed shape: every node v past the initial clique chose exactly m
earlier nodes (its targets, all < v) and was later chosen by some number of later nodes (all > v). So node v's
neighbours split into
  out part: the m targets, a fixed stride, so it needs no row pointers  (12 B/node at m = 3)
  in part:  the neighbours chosen by others, a CSR with row pointers        (~12 B/node at m = 3)
Clique nodes (v < init) have no out part; all their neighbours are in the in part.

Directory layout (all little-endian, raw arrays, no headers, so they can be read in blocks or memory-mapped):
  meta.json   {"format": "polya-split-v1", "n", "m", "init", "n_out", "nnz_in", ...free-form provenance}
  out.i32     int32[(n - init) * m]   row v >= init is out[(v - init) * m : (v - init + 1) * m]
  in_ptr.i64  int64[n + 1]
  in_idx.i32  int32[nnz_in]           each row sorted ascending

The super urn of v is urns[v] + sum(urns[u] for u in out row v) + sum(urns[u] for u in in row v). A neighbour
connected twice (a duplicate edge, weight 2 in the repo's CSR) appears twice.
"""
import json
import os

import numpy as np

FORMAT = "polya-split-v1"


def paths(d):
    return {k: os.path.join(d, k) for k in ("meta.json", "out.i32", "in_ptr.i64", "in_idx.i32")}


def read_meta(d):
    meta = json.load(open(paths(d)["meta.json"]))
    assert meta.get("format") == FORMAT, meta.get("format")
    return meta


def write_meta(d, n, m, init, n_out, nnz_in, **extra):
    meta = {"format": FORMAT, "n": int(n), "m": int(m), "init": int(init), "n_out": int(n_out),
            "nnz_in": int(nnz_in)}
    meta.update(extra)
    json.dump(meta, open(paths(d)["meta.json"], "w"), indent=1)
    return meta


def open_arrays(d, mode="r"):
    """Memory-maps (out, in_ptr, in_idx); nothing is read until used."""
    meta, p = read_meta(d), paths(d)
    out = np.memmap(p["out.i32"], np.int32, mode, shape=(meta["n_out"],)) if meta["n_out"] else np.zeros(0, np.int32)
    in_ptr = np.memmap(p["in_ptr.i64"], np.int64, mode, shape=(meta["n"] + 1,))
    in_idx = np.memmap(p["in_idx.i32"], np.int32, mode, shape=(meta["nnz_in"],))
    return meta, out, in_ptr, in_idx


def from_csr(d, indptr, indices, m, init, data=None, chunk_rows=1 << 22, **extra):
    """
    Writes the split format for a symmetric CSR from the repo's BA growth (rows sorted by column, as scipy or
    csr_kernels.csr_from_edges build them). data, if given, holds integer edge weights (2 = duplicate edge),
    which become repeated entries. Works in row chunks, so the CSR may be a memory map.
    """
    os.makedirs(d, exist_ok=True)
    n = len(indptr) - 1
    p = paths(d)
    n_out = (n - init) * m
    out_f, ptr_f, idx_f = open(p["out.i32"], "wb"), open(p["in_ptr.i64"], "wb"), open(p["in_idx.i32"], "wb")
    nnz_in = 0
    ptr_f.write(np.zeros(1, np.int64).tobytes())
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        k0, k1 = int(indptr[r0]), int(indptr[r1])
        cols = np.asarray(indices[k0:k1], dtype=np.int32)
        deg = np.diff(np.asarray(indptr[r0:r1 + 1], dtype=np.int64))
        if data is not None:
            w = np.asarray(data[k0:k1]).astype(np.int64)
            assert (w >= 1).all() and (w == np.round(w)).all(), "weights must be positive integers"
            rep = np.repeat(np.arange(cols.size), w)
            cols = cols[rep]
            rows_of = np.repeat(np.arange(r0, r1), deg)[rep]
            deg = np.bincount(rows_of - r0, minlength=r1 - r0)
        rows = np.repeat(np.arange(r0, r1, dtype=np.int64), deg)
        lower = (cols < rows) & (rows >= init)
        # out part: neighbours below v for v >= init, exactly m of them
        lo_cnt = np.bincount((rows[lower] - r0), minlength=r1 - r0)
        bad = np.flatnonzero((lo_cnt != m) & (np.arange(r0, r1) >= init))
        assert bad.size == 0, f"rows {bad[:5] + r0} have {lo_cnt[bad[:5]]} lower neighbours, expected m={m}"
        out_f.write(cols[lower].astype(np.int32).tobytes())
        upper = ~lower
        in_cnt = np.bincount(rows[upper] - r0, minlength=r1 - r0)
        idx_f.write(cols[upper].astype(np.int32).tobytes())
        ptr_f.write((nnz_in + np.cumsum(in_cnt)).astype(np.int64).tobytes())
        nnz_in += int(in_cnt.sum())
    for f in (out_f, ptr_f, idx_f):
        f.close()
    return write_meta(d, n, m, init, n_out, nnz_in, **extra)


def validate(d, indptr=None, indices=None, data=None, rows=None, rng_seed=0):
    """
    Structural checks (sizes, monotone in_ptr, ranges, out rows below v, in rows sorted) and, if a reference
    CSR is given, that every sampled row has the same neighbour multiset. Returns a list of problems.
    """
    meta, out, in_ptr, in_idx = open_arrays(d)
    n, m, init = meta["n"], meta["m"], meta["init"]
    errs = []
    if out.size != (n - init) * m:
        errs.append("out size")
    if in_ptr[0] != 0 or in_ptr[-1] != in_idx.size:
        errs.append("in_ptr ends")
    rng = np.random.default_rng(rng_seed)
    if rows is None:
        rows = np.unique(np.concatenate([np.arange(min(n, init + 3)), rng.integers(0, n, 2000), [n - 1]]))
    for v in rows:
        v = int(v)
        o = np.asarray(out[(v - init) * m:(v - init + 1) * m]) if v >= init else np.zeros(0, np.int32)
        i = np.asarray(in_idx[in_ptr[v]:in_ptr[v + 1]])
        if v >= init and not (o < v).all():
            errs.append(f"row {v}: out part not below v")
        if i.size and (np.diff(i) < 0).any():
            errs.append(f"row {v}: in part not sorted")
        if v >= init and i.size and not (i > v).all():
            errs.append(f"row {v}: in part not above v")
        if indptr is not None:
            ref = np.asarray(indices[indptr[v]:indptr[v + 1]])
            if data is not None:
                ref = np.repeat(ref, np.asarray(data[indptr[v]:indptr[v + 1]]).astype(np.int64))
            if not np.array_equal(np.sort(np.concatenate([o, i])), np.sort(ref)):
                errs.append(f"row {v}: neighbours differ from the reference CSR")
        if len(errs) > 20:
            break
    return errs
