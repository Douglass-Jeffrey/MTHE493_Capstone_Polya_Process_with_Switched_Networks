"""
Run the Polya process on a Barabasi-Albert graph with a chosen set of optimizations, and time each phase.

    python run_sim.py --batches 128                                    # 71.6M nodes, every optimization
    python run_sim.py --batches 128 --preset original                  # the step as it was before the work
    python run_sim.py --batches 128 --renumber none                    # a preset with one switch changed
    python run_sim.py --batches 32 --memory 72 --steps 864 --hist hist.csv
    python run_sim.py --batches 128 --backend replicas --runs 8        # 8 runs side by side on one graph
    python run_sim.py --batches 1788 --backend engine --auto-budget    # 10^9 nodes, adjacency outside VRAM

The graph is the benchmark's: a 64-node clique, then `batches` batches each of 8, 128, 2048, 32768 and 524288
new nodes with m = 3, so n = 64 + 559,240 * batches (32: 17.9M, 64: 35.8M, 128: 71.6M, 256: 143.2M,
1788: 10^9, engine only). Every urn starts with one ball of each colour; delta and decay are the same for
every node. Each run appends a summary line (timings, switches, an md5 of the final urns) to work/runs.jsonl.

Backends
  core      the simulator itself (Graph, Barabasi_Albert_Growth, Polya_Process), with the switches below
  replicas  R runs side by side on one graph (scale/replicas.py), so one cache miss serves every run;
            16-bit urns and Philox random numbers, so the runs differ from the core backend's for the same seed
  engine    the out-of-core engine (scale/engine.py) on the same graph, written by scale/gen.py to
            work/data/ on first use and reused after; only the 16-bit urns are kept in VRAM

Core switches (the preset sets each one; a switch given on the command line overrides the preset)
  --kernel          auto | fused | cusparse    the fused gather-and-draw kernel, or the original cuSPARSE product
  --memory-format   auto | bits | int32        finite memory as a ring of bits, or the original int32 array
  --csr-build       auto | gpu | scipy | cupyx where the CSR is built
  --csr-placement   auto | device | host       the CSR column indices in VRAM or in pinned system RAM
  --renumber        none | bfs | rcm | degree | random
                                               node order for the steps; saved urns are mapped back to the
                                               original ids, but each node gets a different random number than
                                               without renumbering, so the urns differ (same distribution)
  --[no-]reserve    size the edge buffer once before growth
  --[no-]free-cache free CuPy's cache of outgrown buffers during growth
  --[no-]keep-edges keep the edge buffer after the CSR build (mid-run node-growth interventions need it)

Presets
  all       every optimization (the default)
  fused     the fused kernel as first committed (optimization_extensions): CSR built by scipy, growth without
            its later fixes, int32 memory, growth order
  original  the code before the optimizations: cuSPARSE product with float64 copies, CSR built by scipy, int32
            memory, CSR in VRAM, edge buffer kept. Growth still reads its stubs from the edge buffer (same
            samples, less memory than the stored stub list), so its memory use is close to, not exactly, the
            original's
"""
import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))

from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process, cp   # noqa: E402
from capstone_optimized.scale.paths import CACHE, DATA, ROOT                           # noqa: E402

SIZES = [8, 128, 2048, 32768, 524288]
SWITCHES = ("kernel", "memory_format", "csr_build", "csr_placement", "renumber", "reserve", "free_cache", "keep_edges")
PRESETS = {
    "all": dict(kernel="auto", memory_format="auto", csr_build="auto", csr_placement="auto", renumber="bfs",
                reserve=True, free_cache=True, keep_edges=False),
    "fused": dict(kernel="fused", memory_format="int32", csr_build="scipy", csr_placement="auto", renumber="none",
                  reserve=False, free_cache=False, keep_edges=False),
    "original": dict(kernel="cusparse", memory_format="int32", csr_build="scipy", csr_placement="device",
                     renumber="none", reserve=False, free_cache=False, keep_edges=True),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["core", "replicas", "engine"], default="core")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="all")
    ap.add_argument("--batches", type=int, default=32, help="batches of each size (default 32: 17.9M nodes)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=493)
    ap.add_argument("--m", type=int, default=3, help="edges per new node")
    ap.add_argument("--init", type=int, default=64, help="nodes in the initial clique")
    ap.add_argument("--memory", type=int, default=0, help="finite memory M in steps (0 = off)")
    ap.add_argument("--delta", type=int, default=1, help="balls added to the drawn colour each step")
    ap.add_argument("--decay", type=int, default=1, help="balls removed when a draw leaves the memory")

    core = ap.add_argument_group("core switches (default: from the preset)")
    core.add_argument("--kernel", choices=Polya_Process.KERNELS)
    core.add_argument("--memory-format", choices=Polya_Process.MEMORY_FORMATS)
    core.add_argument("--csr-build", choices=["auto", "gpu", "scipy", "cupyx"])
    core.add_argument("--csr-placement", choices=["auto", "device", "host"])
    core.add_argument("--renumber", choices=["none", "bfs", "rcm", "degree", "random"])
    core.add_argument("--reserve", action=argparse.BooleanOptionalAction)
    core.add_argument("--free-cache", action=argparse.BooleanOptionalAction)
    core.add_argument("--keep-edges", action=argparse.BooleanOptionalAction)

    rep = ap.add_argument_group("replicas backend")
    rep.add_argument("--runs", type=int, default=8, choices=[1, 2, 4, 8, 16])

    eng = ap.add_argument_group("engine backend")
    eng.add_argument("--graph-dir", help="split-format graph to use (default: work/data/ba_..., generated if missing)")
    eng.add_argument("--placement", default="auto", choices=["auto", "vram", "pinned", "stream", "mixed"])
    eng.add_argument("--block-mb", type=float, default=None, help="size blocks by adjacency bytes (96 at 10^9)")
    eng.add_argument("--block-rows", type=int, default=1 << 22)
    eng.add_argument("--pinned-gb", type=float, default=8.0)
    eng.add_argument("--ram-gb", type=float, default=0.0)
    eng.add_argument("--auto-budget", action="store_true", help="size the pinned and RAM tiers from free RAM")
    eng.add_argument("--cpu-blocks", type=int, default=0, help="blocks drawn by the CPU (the last ones)")
    eng.add_argument("--push-k", type=int, default=0, help="push rows [init, K); -1 = as many as fit in VRAM")
    eng.add_argument("--ring-file", default=None, help="keep the finite-memory ring in this file (10^9 nodes)")

    out = ap.add_argument_group("output")
    out.add_argument("--save-urns", metavar="PATH", help="final urns as .npy, in the original node order")
    out.add_argument("--hist", metavar="PATH", help="per-step histogram (50 bins) of the red fraction, as CSV")
    out.add_argument("--save-csr", metavar="NAME", help="write the CSR (growth order) to work/cache/NAME/")
    out.add_argument("--log", default=os.path.join(ROOT, "runs.jsonl"), help="append the summary here ('' = off)")

    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.backend == "engine" and args.memory and args.cpu_blocks:
        ap.error("the engine does not combine --memory with --cpu-blocks")
    return args


class Timer:
    """Wall time of each phase, with the GPU's queued work finished at both ends."""

    def __init__(self):
        self.t = {}

    @contextlib.contextmanager
    def __call__(self, name):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        yield
        cp.cuda.Device().synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0


def grow(args, timer):
    """The benchmark's BA graph, built as test_benchmark.py builds it (same samples as scale.gen for the same seed)."""
    cp.random.seed(args.seed)
    g = Graph(num_nodes=args.init, num_colors=2, use_gpu=True, node_urns=cp.ones((args.init, 2), dtype=cp.int32))
    g.csr_build, g.csr_placement = args.csr_build, args.csr_placement
    growth = Barabasi_Albert_Growth(g, m=args.m, initial_nodes=args.init)
    growth.free_cache = args.free_cache
    with timer("growth"):
        if args.reserve:
            growth.reserve(args.init + args.batches * sum(SIZES))
        for bs in SIZES:
            for _ in range(args.batches):
                growth.grow_batch(batch_size=bs)
    with timer("csr"):
        growth.finalize_growth(release_edges=not args.keep_edges)
    if args.save_csr:
        save_csr(g, args)
    if args.renumber != "none":
        with timer("renumber"):
            g.renumber(node_order(g, args.renumber, args.seed))
    return g


def node_order(g, how, seed):
    if how == "bfs":
        return g.bfs_order()
    if how == "rcm":
        return g.rcm_order()
    if how == "degree":
        return cp.argsort(-cp.diff(g.csr_indptr)).astype(cp.int32)
    return cp.random.RandomState(seed + 1).permutation(g.num_nodes).astype(cp.int32)   # 'random'


def save_csr(g, args):
    d = os.path.join(CACHE, args.save_csr)
    os.makedirs(d, exist_ok=True)
    indices = g.csr_indices if isinstance(g.csr_indices, np.ndarray) else cp.asnumpy(g.csr_indices)
    np.save(os.path.join(d, "indptr.npy"), cp.asnumpy(g.csr_indptr))
    np.save(os.path.join(d, "indices.npy"), indices)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"n": int(g.num_nodes), "nnz": int(indices.size), "batches": args.batches, "seed": args.seed,
                   "m": args.m, "init": args.init}, f, indent=1)
    print(f"CSR written to {d}")


def to_original_order(g, urns):
    """urns indexed by the renumbered ids -> by the original ids (Graph.node_order[new] = old)."""
    order = getattr(g, "node_order", None)
    if order is None:
        return urns
    out = cp.empty_like(urns)
    out[cp.asarray(order)] = urns
    return out


def red_hist(urns):
    red = urns[..., 0].astype(cp.float64).ravel()
    total = urns.sum(axis=-1).astype(cp.float64).ravel()
    return cp.asnumpy(cp.histogram(red / cp.maximum(total, 1), bins=50, range=(0.0, 1.0))[0])


def timed_steps(step, steps, after=None):
    times = []
    for _ in range(steps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        step()
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
        if after is not None:
            after()
    return times


def run_core(args, timer):
    g = grow(args, timer)
    n = g.num_nodes
    kw = dict(delta=cp.full((n, 2), args.delta, dtype=cp.int32), kernel=args.kernel, memory_format=args.memory_format)
    if args.memory:
        kw.update(memory_enabled=True, memory_decay_time=args.memory,
                  mem_decay=cp.full((n, 2), args.decay, dtype=cp.int32))
    polya = Polya_Process(g, **kw)
    hist = []
    times = timed_steps(polya.step, args.steps, (lambda: hist.append(red_hist(g.node_urns))) if args.hist else None)
    info = {"n": n, "entries": int(g.csr_indptr[-1]), "csr_indices_on": "host" if isinstance(g.csr_indices, np.ndarray) else "device",
            "memory_as": "bits" if getattr(polya, "_mem_bits", None) is not None else ("int32" if args.memory else None)}
    return times, to_original_order(g, g.node_urns), hist, info


def run_replicas(args, timer):
    from capstone_optimized.scale.replicas import ReplicaPolya
    g = grow(args, timer)
    if not isinstance(g.csr_indices, cp.ndarray):
        sys.exit("replicas needs the CSR in VRAM: try --csr-placement device, or a smaller graph")
    g.node_urns = None                                         # the replicas keep their own 16-bit urns
    cp.get_default_memory_pool().free_all_blocks()
    rp = ReplicaPolya(g.csr_indptr, g.csr_indices, args.runs, deltas=[args.delta] * args.runs,
                      decays=[args.decay] * args.runs, memory=args.memory, seed=args.seed)
    hist = []
    times = timed_steps(rp.step, args.steps, (lambda: hist.append(red_hist(rp.urns(0)))) if args.hist else None)
    info = {"n": g.num_nodes, "entries": int(g.csr_indptr[-1]), "runs": args.runs,
            "per_run_step_s": float(np.median(times)) / args.runs, "hist_of": "run 0"}
    return times, to_original_order(g, rp.urns()), hist, info


def run_engine(args, timer):
    import psutil
    from capstone_optimized.scale import gen, split_format as sf
    from capstone_optimized.scale.engine import Engine
    d = args.graph_dir or os.path.join(DATA, f"ba_m{args.m}_i{args.init}_b{args.batches}_s{args.seed}")
    if not os.path.exists(os.path.join(d, "COMPLETE")):
        shutil.rmtree(d, ignore_errors=True)
        print(f"writing the graph to {d} (about 30 GB and 2 minutes at 10^9 nodes)")
        with timer("generate"):
            gen.generate(d, args.batches, seed=args.seed, m=args.m, init=args.init)
        open(os.path.join(d, "COMPLETE"), "w").close()
    meta = sf.read_meta(d)
    if args.auto_budget:                                       # as bench_engine.py: leave RAM for the OS
        avail = psutil.virtual_memory().available / 2**30
        slots_gb = 6 * ((args.block_mb or 128) + 2) / 1024
        fixed = meta["n"] / 2**30 + slots_gb + 1.5 + 3.0
        args.pinned_gb = max(0.0, min(14.5 - meta["n"] / 2**30 - slots_gb, avail - fixed))
        args.ram_gb = max(0.0, avail - fixed - args.pinned_gb)
        print(f"auto budget: {avail:.1f} GiB available -> pinned {args.pinned_gb:.1f} GiB, RAM {args.ram_gb:.1f} GiB")
    with timer("load"):
        eng = Engine(d, placement=args.placement, block_rows=args.block_rows, block_mb=args.block_mb,
                     pinned_budget_gb=args.pinned_gb, ram_budget_gb=args.ram_gb, seed=args.seed, delta=args.delta,
                     push_k=args.push_k)
        if args.cpu_blocks:
            eng.enable_cpu(args.cpu_blocks)
        if args.memory:
            eng.enable_memory(args.memory, decay=args.decay, ring_path=args.ring_file)
    hist = []
    times = timed_steps(eng.step, args.steps, (lambda: hist.append(red_hist(eng.urns))) if args.hist else None)
    info = {"n": eng.n, "graph_dir": d, "placement": eng.placement,
            "blocks": {w: sum(b["where"] == w for b in eng.blocks) for w in ("vram", "pinned", "ram", "stream")},
            "overflow": bool(eng.overflowed())}
    urns = eng.urns.astype(cp.int32)
    eng.close()
    return times, urns, hist, info


def main():
    args = parse_args()
    os.makedirs(ROOT, exist_ok=True)
    timer = Timer()
    t_all = time.perf_counter()
    run = {"core": run_core, "replicas": run_replicas, "engine": run_engine}[args.backend]
    times, urns, hist, info = run(args, timer)
    timer.t["steps"] = float(np.sum(times))
    total = time.perf_counter() - t_all

    urns_h = cp.asnumpy(urns)
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": args.backend, "preset": args.preset,
               "batches": args.batches, "seed": args.seed, "steps": args.steps, "memory": args.memory,
               "delta": args.delta, "decay": args.decay,
               **({k: getattr(args, k) for k in SWITCHES} if args.backend != "engine" else {}),
               **info, "phases_s": {k: round(v, 3) for k, v in timer.t.items()},
               "step_median_s": round(float(np.median(times)), 4) if times else None,
               "total_s": round(total, 2), "urns_md5": hashlib.md5(urns_h.tobytes()).hexdigest()}

    print(f"\n{args.backend} backend, preset {args.preset}, {summary['n']:,} nodes, {args.steps} steps")
    if args.backend != "engine":
        print("  " + ", ".join(f"{k}={getattr(args, k)}" for k in SWITCHES))
    for k, v in timer.t.items():
        print(f"  {k:<9} {v:9.2f} s")
    if times:
        print(f"  step      {np.median(times):9.4f} s median ({min(times):.4f} - {max(times):.4f})")
    print(f"  total     {total:9.2f} s    urns md5 {summary['urns_md5']}")

    if args.save_urns:
        np.save(args.save_urns, urns_h)
        print(f"urns saved to {args.save_urns}")
    if args.hist:
        np.savetxt(args.hist, np.array(hist, dtype=np.int64), delimiter=",", fmt="%d")
        print(f"histograms saved to {args.hist}")
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
