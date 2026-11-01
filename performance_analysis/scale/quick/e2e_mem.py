"""End-to-end run of the repo pipeline (BA growth -> finalize_growth -> Polya steps) with either the original
code (--src orig, HEAD versions of the changed files) or the working tree (--src new). Saves the final urns
so two runs can be compared bit for bit, plus timings."""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", choices=["orig", "new"], required=True)
ap.add_argument("--batches", type=int, default=2)
ap.add_argument("--sizes", default="8,128,2048,32768,524288")
ap.add_argument("--colors", type=int, default=2)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--delta", choices=["default", "random", "random_f"], default="default")
ap.add_argument("--memory", type=int, default=0, help="memory_decay_time, 0 = memory off")
ap.add_argument("--random-urns", action="store_true")
ap.add_argument("--placement", default="auto")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", required=True)
ap.add_argument("--free-pool", action="store_true", help="free CuPy's cached blocks after every growth batch")
ap.add_argument("--ram-guard-gb", type=float, default=0, help="abort if free system RAM drops below this")
args = ap.parse_args()

if args.ram_guard_gb:
    import threading, psutil

    def guard():
        while True:
            free = psutil.virtual_memory().available / 1e9
            if free < args.ram_guard_gb:
                print(f"RAM GUARD: only {free:.1f} GB free, aborting", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=guard, daemon=True).start()
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

D = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\dougl\polya-overnight"
sys.path.insert(0, {"orig": os.path.join(r"C:/Users/dougl/AppData/Local/Temp/claude/c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks/7bf0ea5c-bb18-4a6c-8e60-f701292fffbe/scratchpad", "base_src", "src"), "new": os.path.join(REPO, "src")}[args.src])
from capstone_optimized.cupy_fallback import cp, CUPY_AVAILABLE
from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process
import capstone_optimized.core.graph_x as gx
_norm = lambda p: os.path.normcase(os.path.normpath(p))
assert _norm(os.path.dirname(gx.__file__)).startswith(_norm(sys.path[0])), gx.__file__

sync = cp.cuda.Device().synchronize if CUPY_AVAILABLE else (lambda: None)
cp.random.seed(args.seed)
c, init = args.colors, 64


def init_urns(b_a, bs):
    b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, cp.ones((bs, c), dtype=cp.int32)])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs


g = Graph(num_nodes=init, num_colors=c, use_gpu=True, node_urns=cp.ones((init, c), dtype=cp.int32))
g.csr_placement = args.placement
growth = Barabasi_Albert_Growth(g, m=3, initial_nodes=init)
for bs in [int(s) for s in args.sizes.split(",")]:
    for _ in range(args.batches):
        growth.grow_batch(batch_size=bs, urn_addition_callback=init_urns)
        if args.free_pool:
            cp.get_default_memory_pool().free_all_blocks()
    log(f"grown to {g.num_nodes:,} nodes, {g.num_edges:,} edges")
t0 = time.time()
growth.finalize_growth()
sync()
csr_s = time.time() - t0
log(f"CSR built in {csr_s:.1f}s, indices on host: {isinstance(getattr(g, 'csr_indices', None), np.ndarray)}")
n = g.num_nodes
if args.random_urns:
    g.set_node_urns(cp.random.randint(0, 40, size=(n, c)))
delta = None
if args.delta != "default":
    delta = cp.random.randint(1, 5, size=(n, c)).astype(cp.int32)
    if args.delta == "random_f":
        delta = cp.asfortranarray(delta)
kw = {}
if args.memory:
    kw = dict(memory_enabled=True, memory_decay_time=args.memory, remove_initial_urns=True,
              mem_decay=cp.random.randint(0, 2, size=(n, c)).astype(cp.int32))
p = Polya_Process(g, delta=delta, **kw)
to_np = cp.asnumpy if CUPY_AVAILABLE else np.asarray
mega_check = to_np(g.get_mega_urns()) if n <= 2_000_000 else None
ts = []
for _ in range(args.steps):
    t = time.time()
    p.step()
    sync()
    ts.append(time.time() - t)
urns = cp.asnumpy(g.node_urns) if CUPY_AVAILABLE else g.node_urns
np.save(args.out + "_urns.npy", urns)
if mega_check is not None:
    np.save(args.out + "_mega.npy", mega_check)
info = {"src": args.src, "n": n, "nnz": int(g.num_edges), "csr_s": csr_s, "steps_s": ts,
        "fast": getattr(g, "csr_indices", None) is not None,
        "indices_on_host": isinstance(getattr(g, "csr_indices", None), np.ndarray),
        "cupy": CUPY_AVAILABLE}
json.dump(info, open(args.out + ".json", "w"))
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items() if k != "steps_s"}),
      f"median step {np.median(ts) * 1e3:.1f} ms")
