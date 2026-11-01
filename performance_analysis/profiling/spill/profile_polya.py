"""
Profiling harness for the BA growth + Polya benchmark (mirrors test_benchmark.run_benchmark).

Uses the project code unchanged. Adds NVTX ranges for Nsight Systems, per-step timing,
memory snapshots, and a "residency probe" that times a full read of each persistent array:
~200+ GB/s means the array is in GDDR6 VRAM, ~10 GB/s or less means the GPU is reading it
from system RAM over PCIe (WDDM sysmem fallback).

--fix options (applied by monkeypatching, the repo is not modified):
  none   : as benchmarked
  free   : drop graph.edges and growth.stubs after the CSR is built
  f32    : multiply the float32 CSR by float32 urns, so cuSPARSE does not upcast A to float64
  both   : free + f32
"""
import argparse, json, os, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
WORK = os.environ.get("POLYA_WORK") or os.path.join(REPO, "work")   # as src/capstone_optimized/scale/paths.py
ORIG_SRC = os.path.join(WORK, "orig_src")   # the code at fd79e4d, which these measurements were taken on
sys.path.insert(0, ORIG_SRC)

from capstone_optimized import cupy_fallback as cf
cp = cf.cp
from cupy.cuda import nvtx
from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=64)
ap.add_argument("--sizes", default="8,128,2048,32768,524288")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--fix", default="none", choices=["none", "free", "f32", "both"])
ap.add_argument("--probe", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

MB = 1024 ** 2
pool = cp.get_default_memory_pool()
log = {"args": vars(args), "snapshots": {}, "probe": {}, "steps": []}


def snap(tag):
    free, total = cp.cuda.runtime.memGetInfo()
    s = {"gpu_free_mb": free / MB, "gpu_total_mb": total / MB,
         "pool_used_mb": pool.used_bytes() / MB, "pool_total_mb": pool.total_bytes() / MB}
    log["snapshots"][tag] = s
    print(f"[{tag}] free={s['gpu_free_mb']:.0f} MB  pool used={s['pool_used_mb']:.0f} MB  "
          f"pool held={s['pool_total_mb']:.0f} MB", flush=True)


def read_bw(name, arr, reps=3):
    """Time a full-array read (sum reduction). Returns list of GB/s per rep."""
    if arr is None:
        return
    # integer view so the reduction is never limited by the 1660 Ti's 1/32-rate FP64
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
        ms = cp.cuda.get_elapsed_time(ev0, ev1)
        out.append(flat.nbytes / (ms / 1e3) / 1e9)
    print(f"  probe {name:<14} {flat.nbytes / MB:8.0f} MB  " +
          "  ".join(f"{g:7.1f} GB/s" for g in out), flush=True)
    return {"mb": flat.nbytes / MB, "gbps": out}


def warm_up(seconds=1.5):
    # Idle phases drop the 1660 Ti into a low P-state with downclocked GDDR6 (~12 GB/s),
    # which looks just like PCIe. Keep the GPU busy on the VRAM reference buffer so the
    # clocks are back up before measuring.
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
    A = graph.adj_matrix
    arrays = {"vram_reference": REF, "edges_buffer": graph.edges, "stubs": growth.stubs,
              "csr.data": A.data, "csr.indices": A.indices, "csr.indptr": A.indptr,
              "node_urns": graph.node_urns, "deltas": deltas}
    log["probe"][tag] = {k: read_bw(k, v) for k, v in arrays.items() if v is not None}


def probe_spmm(graph):
    """Rebuild the operands a Polya step hands to cuSPARSE (same sizes and order, so the pool
    hands back the same cached blocks), probe where they live, and time SpMM by itself."""
    from cupyx import cusparse
    print("SpMM operand probe", flush=True)
    warm_up()
    A = graph.adj_matrix
    nvtx.RangePush("instrumented step: cast operands")
    if args.fix in ("f32", "both"):
        a_op, b_op = A, cp.asfortranarray(graph.node_urns.astype(cp.float32))
    else:  # what cusparse._cast_common_type does to float32 CSR @ int32 urns
        b_op = cp.asfortranarray(graph.node_urns)
        a_op, b_op = A.astype(cp.float64), b_op.astype(cp.float64)
    cp.cuda.Device().synchronize()
    nvtx.RangePop()
    res = {}
    if a_op is not A:
        res["A64.data"] = read_bw("A64.data", a_op.data)
        res["A64.indices"] = read_bw("A64.indices", a_op.indices)
        res["A64.indptr"] = read_bw("A64.indptr", a_op.indptr)
    res["B_dense"] = read_bw("B_dense", b_op)
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    t = []
    for r in range(2):
        nvtx.RangePush(f"instrumented step: spmm #{r}")
        ev0.record()
        c = cusparse.spmm(a_op, b_op)
        ev1.record()
        ev1.synchronize()
        nvtx.RangePop()
        t.append(cp.cuda.get_elapsed_time(ev0, ev1) / 1e3)
        if r == 0:
            res["C_out"] = read_bw("C_out", c)
        del c
    res["spmm_s"] = t
    print(f"  spmm alone: " + ", ".join(f"{x:.3f}s" for x in t), flush=True)
    log["probe"]["spmm_operands"] = res


def init_urns(b_a, bs):
    added = cp.zeros((bs, b_a.graph.num_colors), dtype=cp.int32)
    added[:, 0] = 1
    added[:, 1] = 1
    b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, added])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs


if args.fix in ("f32", "both"):
    def get_mega_urns_f32(self, include_self=True):
        if self.adj_matrix is None:
            return self.node_urns.copy()
        mega = self.adj_matrix @ self.node_urns.astype(cp.float32)
        if include_self:
            mega += self.node_urns
        return mega
    Graph.get_mega_urns = get_mega_urns_f32

sizes = [int(s) for s in args.sizes.split(",")]
initial_nodes, num_colors = 64, 2
snap("start")
# 64 MB allocated first, so it is certainly in VRAM: the bandwidth yardstick for the probe
REF = cp.ones(16 * MB, dtype=cp.int32) if args.probe else None

urns0 = cp.zeros((initial_nodes, num_colors), dtype=cp.int32)
urns0[:, :] = 1
graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True, node_urns=urns0)
growth = Barabasi_Albert_Growth(graph, m=3, initial_nodes=initial_nodes)

t0 = time.time()
nvtx.RangePush("BA growth")
for bs in sizes:
    for _ in range(args.batches):
        growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
cp.cuda.Device().synchronize()
nvtx.RangePop()
t1 = time.time()
snap("after_growth")

nvtx.RangePush("CSR build")
growth.finalize_growth()
cp.cuda.Device().synchronize()
nvtx.RangePop()
t2 = time.time()
snap("after_csr")
log["nodes"] = int(graph.num_nodes)
log["edges"] = int(graph.num_edges)
log["growth_s"] = t1 - t0
log["csr_s"] = t2 - t1
print(f"nodes={graph.num_nodes:,} edges={graph.num_edges:,} growth={t1 - t0:.1f}s csr={t2 - t1:.1f}s", flush=True)

if args.fix in ("free", "both"):
    graph.edges = None
    growth.stubs = None
    pool.free_all_blocks()
    snap("after_free")

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
    probe_spmm(graph)

if args.out:
    with open(args.out, "w") as f:
        json.dump(log, f, indent=1)
