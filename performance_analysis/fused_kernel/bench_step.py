"""Time one Polya step (rand + fused gather/draw + urn update) on a cached CSR.

--mode resident : indptr, indices, urns, delta, rand, draws all in VRAM
--mode zerocopy : indices in pinned host RAM, read by the kernel directly over PCIe (sequential reads)
--mode stream   : indices in pinned host RAM, copied to two VRAM chunk buffers on a copy stream while the
                  previous chunk is processed on a compute stream
--mode orig     : the repo's Polya_Process.step (cupyx SpMM), for reference
"""
import argparse, json, os, sys, time
import numpy as np
import cupy as cp
from cupy.cuda import nvtx
import kern

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("--mode", default="resident", choices=["resident", "zerocopy", "stream", "orig"])
ap.add_argument("--lanes", default="1,2,4,8,16,32")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--chunk-mb", type=int, default=128)
ap.add_argument("--tag", default="")
args = ap.parse_args()

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
ORIG_SRC = os.path.join(WORK, "orig_src")   # the code at fd79e4d, which these measurements were taken on
cdir = os.path.join(WORK, "cache", args.cache)
pool = cp.get_default_memory_pool()
GB = 1e9

indptr_h = np.load(os.path.join(cdir, "indptr.npy"))
n = indptr_h.size - 1
nnz = int(indptr_h[-1])
indptr = cp.asarray(indptr_h)
urns = cp.ones((n, 2), cp.int32)
delta = cp.ones((n, 2), cp.int32)
two = np.int64(2)

if args.mode in ("resident", "orig"):
    indices = cp.asarray(np.load(os.path.join(cdir, "indices.npy"), mmap_mode="r"))
else:
    pinned = cp.cuda.alloc_pinned_memory(nnz * 4)
    indices_h = np.frombuffer(pinned, np.int32, nnz)
    indices_h[:] = np.load(os.path.join(cdir, "indices.npy"), mmap_mode="r")

if args.mode == "stream":
    ch = args.chunk_mb * 2 ** 20 // 4                       # indices per chunk
    bounds = [0]
    while bounds[-1] < n:                                   # row boundaries, ~ch indices per chunk
        r = int(np.searchsorted(indptr_h, indptr_h[bounds[-1]] + ch, side="right")) - 1
        bounds.append(min(n, max(r, bounds[-1] + 1)))
    maxk = max(int(indptr_h[b] - indptr_h[a]) for a, b in zip(bounds[:-1], bounds[1:]))
    bufs = [cp.empty(maxk, cp.int32) for _ in range(2)]
    s_copy, s_comp = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    ev_copied = [cp.cuda.Event() for _ in range(2)]
    ev_done = [cp.cuda.Event() for _ in range(2)]
    for e in ev_done:
        e.record(s_comp)

if args.mode == "orig":
    sys.path.insert(0, ORIG_SRC)   # the repo's step as it was (cupyx SpMM)
    from capstone_optimized.core import Graph, Polya_Process
    from cupyx.scipy.sparse import csr_matrix
    g = Graph(num_nodes=n, num_colors=2, use_gpu=True, node_urns=urns)
    g.adj_matrix = csr_matrix((cp.ones(nnz, cp.float32), indices, indptr), shape=(n, n))
    proc = Polya_Process(g, delta=delta)


def draw_phase(rnd, draws, lanes):
    if args.mode == "resident":
        kern.draw2(indptr, indices, urns, rnd, draws, lanes=lanes)
    elif args.mode == "zerocopy":
        kern.draw2(indptr, np.uint64(pinned.ptr), urns, rnd, draws, lanes=lanes)
    else:
        cur = cp.cuda.get_current_stream()
        ev = cp.cuda.Event()
        ev.record(cur)
        s_comp.wait_event(ev)                               # rand must exist before the first chunk
        for c, (r0, r1) in enumerate(zip(bounds[:-1], bounds[1:])):
            b = c % 2
            k0, k1 = int(indptr_h[r0]), int(indptr_h[r1])
            s_copy.wait_event(ev_done[b])                   # buffer free again?
            cp.cuda.runtime.memcpyAsync(bufs[b].data.ptr, pinned.ptr + 4 * k0, 4 * (k1 - k0),
                                        cp.cuda.runtime.memcpyHostToDevice, s_copy.ptr)
            ev_copied[b].record(s_copy)
            s_comp.wait_event(ev_copied[b])
            kern.draw2(indptr, bufs[b], urns, rnd, draws, lanes=lanes, row0=r0, nrows=r1 - r0,
                       idx_base=k0, stream=s_comp)
            ev_done[b].record(s_comp)
        ev2 = cp.cuda.Event()
        ev2.record(s_comp)
        cur.wait_event(ev2)


def step(lanes):
    if args.mode == "orig":
        proc.step()
        return
    rnd = cp.random.rand(n)
    draws = cp.empty(n, cp.int32)
    draw_phase(rnd, draws, lanes)
    kern.apply_draws(draws, delta, two, urns)


lanes_list = [0] if args.mode == "orig" else [int(x) for x in args.lanes.split(",")]
free, total = cp.cuda.runtime.memGetInfo()
print(f"{args.cache} {args.mode}: n={n:,} nnz={nnz:,} pool used {pool.used_bytes() / GB:.2f} GB, "
      f"device free {free / GB:.2f} GB" + (f", {len(bounds) - 1} chunks" if args.mode == "stream" else ""), flush=True)
for lanes in lanes_list:
    for _ in range(2):
        step(lanes)                                          # warm-up: JIT, pool, clocks
    cp.cuda.Device().synchronize()
    ts = []
    for s in range(args.steps):
        nvtx.RangePush(f"{args.mode} lanes={lanes} step {s}")
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        step(lanes)
        e1.record()
        e1.synchronize()
        nvtx.RangePop()
        ts.append(cp.cuda.get_elapsed_time(e0, e1) / 1e3)
    med = float(np.median(ts))
    rec = {"cache": args.cache, "mode": args.mode, "lanes": lanes, "n": n, "nnz": nnz, "tag": args.tag,
           "step_s": ts, "median_s": med, "chunk_mb": args.chunk_mb if args.mode == "stream" else None,
           "pool_used_gb": pool.used_bytes() / GB}
    print(f"  lanes={lanes:>2}: median {med * 1e3:8.1f} ms/step  (min {min(ts) * 1e3:.1f}, max {max(ts) * 1e3:.1f})",
          flush=True)
    with open(os.path.join(D, "results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
