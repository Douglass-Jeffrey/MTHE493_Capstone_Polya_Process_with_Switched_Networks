"""
Sequential vs random-gather bandwidth for a buffer in VRAM and the same buffer pushed into system RAM.

A ballast allocation fills free VRAM first, so the WDDM sysmem fallback places the next
allocation in system RAM. The gather reads 8-byte float64 values at random row indices,
the access pattern cuSPARSE uses to read U[j] for every neighbour j in SpMM.
"""
import json, sys, time
import cupy as cp

MB = 1024 ** 2
pool = cp.get_default_memory_pool()
N_IDX = 50_000_000
idx_cache = {}


def warm(buf):
    t = time.time()
    while time.time() - t < 1.0:
        buf[: 16 * MB // 8].sum()
        cp.cuda.Device().synchronize()


def timed(fn, reps=3):
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    best = 1e9
    for _ in range(reps):
        e0.record(); fn(); e1.record(); e1.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(e0, e1) / 1e3)
    return best


def measure(span_mb, place):
    n = span_mb * MB // 8
    ballast = None
    if place == "sysram":
        free, _ = cp.cuda.runtime.memGetInfo()
        ballast = cp.empty(free + 256 * MB, dtype=cp.uint8)   # overfill VRAM
        ballast[::4096] = 1
    buf = cp.ones(n, dtype=cp.float64)
    idx = cp.random.randint(0, n, size=N_IDX, dtype=cp.int32)
    out = cp.empty(N_IDX, dtype=cp.float64)
    warm(idx.view(cp.float64) if place == "sysram" else buf)
    seq = timed(lambda: buf.sum())
    gat = timed(lambda: cp.take(buf, idx, out=out))
    r = {"span_mb": span_mb, "place": place, "seq_gbps": n * 8 / seq / 1e9,
         "gather_ns": gat / N_IDX * 1e9, "gathers_per_s": N_IDX / gat}
    print(f"{place:>6} {span_mb:6d} MB  sequential {r['seq_gbps']:6.1f} GB/s   random gather "
          f"{r['gather_ns']:6.2f} ns each ({N_IDX * 8 / gat / 1e9:5.2f} GB/s useful)", flush=True)
    del buf, idx, out, ballast
    pool.free_all_blocks()
    return r


res = []
for place in ("vram", "sysram"):
    for span in (64, 256, 512, 1024, 2048):
        if place == "vram" and span > 2048:
            continue
        res.append(measure(span, place))
json.dump(res, open(sys.argv[1], "w"), indent=1)
