"""
Generates one graph with scale.gen and appends a line to results.md.

    python run_gen.py <batches> <name> [--seed 493] [--dev-cache-gb 3] [--validate] [--ready]

Writes work/data/<name>/ (see scale/paths.py). --validate runs split_format.validate plus degree-sum checks;
--ready creates <name>/READY when everything passed (used for ba_1b).
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

from capstone_optimized.scale import gen, split_format as sf

from capstone_optimized.scale.paths import DATA
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.md")
GB = 2**30


def degree_checks(d):
    """
    Whole-graph checks on the GPU (deg array: 4 B per node): every out entry v -> t has t < v; in_idx values in
    range; entries = 2 * edges; diff(in_ptr) = in-degree counted from the out part + clique; and for sampled rows
    every in-row entry u >= init really chose that row.
    """
    import cupy as cp
    meta, out, in_ptr, in_idx = sf.open_arrays(d)
    n, m, init = meta["n"], meta["m"], meta["init"]
    errs = []
    edges = (n - init) * m + init * (init - 1) // 2
    if out.size + in_idx.size != 2 * edges:
        errs.append(f"entries {out.size + in_idx.size} != 2 * edges {2 * edges}")
    deg = cp.zeros(n, cp.int32)
    count = gen._k("count_in")
    step = 1 << 24
    for a in range(0, out.size, step):
        c = cp.asarray(out[a:a + step])
        v = init + (a + cp.arange(c.size, dtype=cp.int64)) // m
        if not bool(((c >= 0) & (c < v)).all()):
            errs.append(f"out entries not in [0, v) in chunk {a}")
        count((24 * 32,), (256,), (c, np.int64(c.size), deg))
        del c, v
    deg[:init] += init - 1
    for a in range(0, n, step):
        b = min(n, a + step)
        p = cp.asarray(in_ptr[a:b + 1])
        if not bool((cp.diff(p) == deg[a:b]).all()):
            errs.append(f"in_ptr row lengths differ from the out-part in-degrees in rows {a}..{b}")
            break
    del deg
    for a in range(0, in_idx.size, step):
        c = cp.asarray(in_idx[a:a + step])
        if int(c.min()) < 0 or int(c.max()) >= n:
            errs.append(f"in_idx out of range in chunk {a}")
            break
    cp.get_default_memory_pool().free_all_blocks()
    rng = np.random.default_rng(1)
    for v in np.concatenate([np.arange(4), rng.integers(init, n, 500)]):
        u = np.asarray(in_idx[in_ptr[v]:in_ptr[v + 1]]).astype(np.int64)   # (u - init) * m overflows int32
        u = u[u >= init]
        if u.size:
            chosen = np.stack([np.asarray(out[(u - init) * m + j]) for j in range(m)], axis=1)
            if not (chosen == v).any(axis=1).all():
                errs.append(f"in row {v} lists a node that did not choose {v}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", type=int)
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=493)
    ap.add_argument("--dev-cache-gb", type=float, default=3.0)
    ap.add_argument("--key-bytes", type=int, default=30, help="VRAM budget per in-part entry in a fill pass")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ready", action="store_true")
    ap.add_argument("--delete", action="store_true", help="delete the graph afterwards")
    ap.add_argument("--validate-only", action="store_true", help="validate an existing graph, no generation")
    args = ap.parse_args()
    d = os.path.join(DATA, args.name)
    if args.validate_only:
        t = time.time()
        errs = sf.validate(d) + degree_checks(d)
        print(f"validation ({time.time() - t:.0f} s): {'valid' if not errs else errs[:5]}", flush=True)
        if args.ready and not errs:
            open(os.path.join(d, "READY"), "w").close()
            print("READY written", flush=True)
        status = ("valid" + (", READY" if args.ready else "")) if not errs else "INVALID: " + "; ".join(errs[:3])
        with open(RESULTS, "a") as f:
            f.write(f"| {args.name} | re-validation | | | | | | | | | | {status} |\n")
        sys.exit(0 if not errs else 1)
    if os.path.exists(d):
        shutil.rmtree(d)
    free = shutil.disk_usage(DATA).free
    n = 64 + args.batches * sum(gen.SIZES)
    need = n * 3 * 4 * 2 + (n + 1) * 8
    print(f"disk free {free / GB:.1f} GB, need ~{need / GB:.1f} GB", flush=True)
    if need > free - 4 * GB:
        sys.exit("not enough disk space")
    meta = gen.generate(d, args.batches, seed=args.seed, dev_cache_gb=args.dev_cache_gb, key_bytes=args.key_bytes)
    line = (f"| {args.name} | {meta['n']:,} | {meta['timings']['growth_s']} | {meta['timings']['in_count_s']} | "
            f"{meta['timings']['in_fill_s']} ({meta['timings']['in_passes']} passes) | "
            f"{meta['timings']['out_write_wait_s']} | {meta['timings']['total_s']} | {meta['memory']['peak_rss_gb']} | "
            f"{meta['memory']['min_avail_ram_gb']} | {meta['memory']['min_vram_free_gb']} | {meta['disk_gb']} |")
    status = ""
    if args.validate:
        t = time.time()
        errs = sf.validate(d) + degree_checks(d)
        status = "valid" if not errs else "INVALID: " + "; ".join(errs[:3])
        print(f"validation ({time.time() - t:.0f} s): {status}", flush=True)
        if args.ready and not errs:
            open(os.path.join(d, "READY"), "w").close()
            status += ", READY"
    with open(RESULTS, "a") as f:
        f.write(line + f" {status} |\n")
    print(line, status, flush=True)
    if args.delete:
        shutil.rmtree(d)


if __name__ == "__main__":
    main()
