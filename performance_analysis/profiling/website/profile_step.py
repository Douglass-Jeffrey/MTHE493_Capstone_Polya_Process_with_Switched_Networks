"""
Profiling harness for BA growth + Polya steps, same structure and NVTX ranges as prof/profile_polya.py
("BA growth", "CSR build", "Polya step k", "probe ..."), but able to run either code version:

  --src orig : the code at fd79e4d (work/orig_src), cuSPARSE A @ U
  --src new  : the repo working tree (branch polya-step-optimization), fused CSR kernels

The residency probe times a full read of each persistent GPU array: ~250 GB/s means VRAM,
~13 GB/s means the driver left it in system RAM. With --src new it never touches graph.adj_matrix,
which would build the float32 matrix lazily.
"""
import argparse, json, os, sys, threading, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", choices=["orig", "new", "exp"], required=True)
ap.add_argument("--batches", type=int, default=64)
ap.add_argument("--sizes", default="8,128,2048,32768,524288")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--free-pool", action="store_true", help="free CuPy's cached blocks after every growth batch")
ap.add_argument("--free-pool-after-growth", action="store_true",
                help="drop CuPy's cached blocks before the CSR build (host RAM headroom under the profiler)")
ap.add_argument("--ram-guard-gb", type=float, default=1.5, help="abort if free system RAM drops below this")
ap.add_argument("--checksum", action="store_true", help="md5 of the final urns and the CSR arrays")
ap.add_argument("--reserve", action="store_true", help="growth.reserve(final nodes) first (exp only)")
ap.add_argument("--renumber", action="store_true", help="breadth-first renumbering after the CSR build (exp only)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
ORIG_SRC = os.path.join(WORK, "orig_src")   # the code at fd79e4d, which these measurements were taken on
D = os.path.dirname(os.path.abspath(__file__))
SRC = {"orig": ORIG_SRC, "new": os.path.join(REPO, "src"),
       "exp": os.path.join(WORK, "base_src", "src")}[args.src]
sys.path.insert(0, SRC)

from capstone_optimized import cupy_fallback as cf
cp = cf.cp
from cupy.cuda import nvtx
from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process
import capstone_optimized.core.graph_x as gx
assert os.path.dirname(gx.__file__).startswith(SRC), gx.__file__

if args.ram_guard_gb:
    import psutil

    def guard():
        while True:
            if psutil.virtual_memory().available / 1e9 < args.ram_guard_gb:
                print("RAM GUARD: aborting", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()

MB = 1024 ** 2
pool = cp.get_default_memory_pool()
log = {"args": vars(args), "snapshots": {}, "probe": {}, "steps": []}
cp.random.seed(args.seed)


def snap(tag):
    free, total = cp.cuda.runtime.memGetInfo()
    s = {"gpu_free_mb": free / MB, "gpu_total_mb": total / MB,
         "pool_used_mb": pool.used_bytes() / MB, "pool_total_mb": pool.total_bytes() / MB}
    log["snapshots"][tag] = s
    print(f"[{tag}] free={s['gpu_free_mb']:.0f} MB  pool used={s['pool_used_mb']:.0f} MB  "
          f"pool held={s['pool_total_mb']:.0f} MB", flush=True)


def read_bw(name, arr, reps=3):
    if arr is None or not isinstance(arr, cp.ndarray):
        return None
    flat = arr.ravel(order="K").view({4: cp.int32, 8: cp.int64}[arr.dtype.itemsize])
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    out = []
    for r in range(reps):
        nvtx.RangePush(f"probe {name} #{r}")
        ev0.record()
        flat.sum(dtype=cp.int64)
        ev1.record()
        ev1.synchronize()
        nvtx.RangePop()
        out.append(flat.nbytes / (cp.cuda.get_elapsed_time(ev0, ev1) / 1e3) / 1e9)
    print(f"  probe {name:<14} {flat.nbytes / MB:8.0f} MB  " + "  ".join(f"{g:7.1f} GB/s" for g in out), flush=True)
    return {"mb": flat.nbytes / MB, "gbps": out}


def warm_up(seconds=1.5):
    # an idle 1660 Ti drops to a low P-state whose GDDR6 bandwidth looks like PCIe; spin it back up first
    nvtx.RangePush("warm-up")
    t = time.time()
    while time.time() - t < seconds:
        for _ in range(20):
            REF.sum(dtype=cp.int64)
        cp.cuda.Device().synchronize()
    nvtx.RangePop()


def probe(tag, graph, growth, deltas):
    print(f"residency probe ({tag})", flush=True)
    warm_up()
    arrays = {"vram_reference": REF, "edges_buffer": graph.edges, "stubs": growth.stubs,
              "node_urns": graph.node_urns, "deltas": deltas}
    if args.src == "orig":
        A = graph.adj_matrix
        arrays.update({"csr.data": A.data, "csr.indices": A.indices, "csr.indptr": A.indptr})
    else:
        arrays.update({"csr.indices": graph.csr_indices, "csr.indptr": graph.csr_indptr})
    log["probe"][tag] = {k: v for k, v in ((k, read_bw(k, a)) for k, a in arrays.items()) if v is not None}


def init_urns(b_a, bs):
    added = cp.ones((bs, b_a.graph.num_colors), dtype=cp.int32)
    b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, added])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs


sizes = [int(s) for s in args.sizes.split(",")]
initial_nodes, num_colors = 64, 2
snap("start")
REF = cp.ones(16 * MB, dtype=cp.int32) if args.probe else None   # 64 MB allocated first, so certainly in VRAM

graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True,
              node_urns=cp.ones((initial_nodes, num_colors), dtype=cp.int32))
growth = Barabasi_Albert_Growth(graph, m=3, initial_nodes=initial_nodes)

t0 = time.time()
nvtx.RangePush("BA growth")
if args.reserve:
    print("reserved:", growth.reserve(initial_nodes + args.batches * sum(sizes)), flush=True)
for bs in sizes:
    for _ in range(args.batches):
        growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
        if args.free_pool:
            pool.free_all_blocks()
cp.cuda.Device().synchronize()
nvtx.RangePop()
t1 = time.time()
snap("after_growth")
if args.free_pool_after_growth:
    pool.free_all_blocks()
    snap("after_free")

nvtx.RangePush("CSR build")
growth.finalize_growth()
cp.cuda.Device().synchronize()
nvtx.RangePop()
t2 = time.time()
snap("after_csr")
log.update(nodes=int(graph.num_nodes), edges=int(graph.num_edges), growth_s=t1 - t0, csr_s=t2 - t1,
           indices_on_host=args.src != "orig" and not isinstance(graph.csr_indices, cp.ndarray))
print(f"nodes={graph.num_nodes:,} edges={graph.num_edges:,} growth={t1 - t0:.1f}s csr={t2 - t1:.1f}s "
      f"indices_on_host={log['indices_on_host']}", flush=True)

if args.renumber:
    t3 = time.time()
    nvtx.RangePush("renumber")
    graph.renumber(graph.bfs_order())
    cp.cuda.Device().synchronize()
    nvtx.RangePop()
    log["renumber_s"] = time.time() - t3
    print(f"renumber {log['renumber_s']:.1f}s", flush=True)
deltas = cp.full((graph.num_nodes, graph.num_colors), 1, dtype=cp.int32)
polya = Polya_Process(graph, memory_enabled=False, memory_decay_time=72, delta=deltas)

if args.probe:
    probe("before_polya", graph, growth, deltas)

for k in range(args.steps):
    nvtx.RangePush(f"Polya step {k}")
    ts = time.time()
    polya.step()
    cp.cuda.Device().synchronize()
    dt = time.time() - ts
    nvtx.RangePop()
    log["steps"].append(dt)
    free, _ = cp.cuda.runtime.memGetInfo()
    print(f"step {k}: {dt:.3f}s  free={free / MB:.0f} MB  pool used={pool.used_bytes() / MB:.0f} MB "
          f"held={pool.total_bytes() / MB:.0f} MB", flush=True)
snap("after_polya")

if args.probe:
    probe("after_polya", graph, growth, deltas)

if args.checksum:
    import hashlib
    h = lambda a: hashlib.md5((a if isinstance(a, np.ndarray) else cp.asnumpy(a)).tobytes()).hexdigest()[:12]
    if args.src == "orig":
        A = graph.adj_matrix
        log["md5"] = {"urns": h(graph.node_urns), "indptr": h(A.indptr), "indices": h(A.indices)}
    else:
        log["md5"] = {"urns": h(graph.node_urns), "indptr": h(graph.csr_indptr), "indices": h(graph.csr_indices)}
    print("md5", log["md5"], flush=True)

if args.out:
    with open(args.out, "w") as f:
        json.dump(log, f, indent=1)
