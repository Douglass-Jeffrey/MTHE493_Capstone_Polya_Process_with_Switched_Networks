import sys
import os
import gc
import time
import csv
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available, CPU memory tracking disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from capstone_optimized import cupy_fallback as cf
    cp_module = getattr(cf, 'cp', None)
    CUPY_AVAILABLE = getattr(cf, 'CUPY_AVAILABLE', False)
    print('cupy_fallback: cp module =', getattr(cp_module, '__name__', None), 'CUPY_AVAILABLE=', CUPY_AVAILABLE)
except Exception as _e:
    print('Could not import capstone_optimized.cupy_fallback:', _e)
    CUPY_AVAILABLE = False

from capstone_optimized.core import (
    Graph, Polya_Process, Switch_Network_Growth,
    Switched_Network_Optimized_Graph, Dim_Optimized_Switch_Network_Growth,
    Barabasi_Albert_Growth, Switch_Network_Node_Growth_Intervention
)

if not CUPY_AVAILABLE:
    print('\nGPU requested but CuPy/CUDA not available in this Python environment.')
    print('Make sure you run this script inside your virtualenv where CuPy is installed:')
    print("  . .venv\\Scripts\\Activate")
    print("  $env:PYTHONPATH = 'src'")
    print("  python src\\capstone_optimized\\experiments\\test_y.py")


# ---------------------------------------------------------------------------
# Peak RSS sampler — runs in a background thread so we capture the true
# high-water mark for the current benchmark, not the process lifetime peak.
# ---------------------------------------------------------------------------
class PeakMemorySampler:
    """
    Samples process RSS every `interval` seconds on a background thread.
    Call start() before the work begins and stop() when it ends.
    peak_mb returns the maximum RSS observed since start().
    """
    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self._peak = 0.0
        self._running = False
        self._thread = None
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process()

    def _sample(self) -> float:
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            return self._process.memory_info().rss / 1024 ** 2
        except Exception:
            return 0.0

    def start(self):
        self._peak = self._sample()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            v = self._sample()
            if v > self._peak:
                self._peak = v
            time.sleep(self._interval)

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # one final sample in case the thread missed the very last spike
        v = self._sample()
        if v > self._peak:
            self._peak = v
        return self._peak

    @property
    def peak_mb(self) -> float:
        return self._peak


def _is_oom_exception(exc):
    msg = str(exc).lower()
    if isinstance(exc, MemoryError):
        return True
    if 'out of memory' in msg or 'no memory' in msg:
        return True
    if 'cuda' in msg and 'memory' in msg:
        return True
    return False


def run_growth_batches(num_batches, batch_size, callback=None):
    for b in range(num_batches):
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                growth.grow_batch(batch_size=batch_size, urn_addition_callback=callback)
                t1 = time.time()
                free_mb = total_mb = None
                if graph.use_gpu and CUPY_AVAILABLE and cp_module is not None:
                    try:
                        free, total = cp_module.cuda.runtime.memGetInfo()
                        free_mb = free / 1024 ** 2
                        total_mb = total / 1024 ** 2
                    except Exception:
                        free_mb = total_mb = None
                break
            except Exception as e:
                if _is_oom_exception(e):
                    if batch_size <= 1:
                        print('Batch size reduced to 1 but still OOM; aborting')
                        raise
                    new_bs = max(1, batch_size // 2)
                    print(f'OOM on batch (size={batch_size}). Reducing to {new_bs}, retrying (attempt {attempt}).')
                    batch_size = new_bs
                    continue
                else:
                    raise


def init_urns(b_a, bs):
    added_node_urns = cp_module.zeros((bs, b_a.graph.num_colors), dtype=cp_module.int32)
    added_node_urns[:, 0] = 1
    added_node_urns[:, 1] = 1
    b_a.graph.node_urns = cp_module.vstack([b_a.graph.node_urns, added_node_urns])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs


def get_gpu_snapshot():
    """Return (gpu_free_mb, gpu_used_pool_mb) or (None, None) if no GPU."""
    if not (CUPY_AVAILABLE and cp_module is not None):
        return None, None
    gpu_free = None
    gpu_used_pool = None
    try:
        free, _ = cp_module.cuda.runtime.memGetInfo()
        gpu_free = free / 1024 ** 2
    except Exception:
        pass
    try:
        mempool = cp_module.get_default_memory_pool()
        gpu_used_pool = mempool.used_bytes() / 1024 ** 2
    except Exception:
        pass
    return gpu_free, gpu_used_pool


def get_rss_mb():
    """Return current process RSS in MB, or None."""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return None


def run_benchmark(initial_nodes, num_batches_array, batch_sizes_array,
                  polya_steps_value, output_csv):
    global graph, growth

    num_colors = 2
    barabasi_num_connections = 3
    memory_enabled = False
    memory_decay_time = 72
    delta_gain = 1

    if len(num_batches_array) != len(batch_sizes_array):
        print("Error: num_batches_array and batch_sizes_array must have the same length")
        return None

    num_batches_list = [int(x) for x in num_batches_array]
    batch_sizes_list = [int(x) for x in batch_sizes_array]

    if any(n <= 0 for n in num_batches_list) or any(b <= 0 for b in batch_sizes_list):
        print("Invalid config: num_batches and batch_sizes must be > 0")
        return None

    expected_final_nodes = initial_nodes + sum(
        n * b for n, b in zip(num_batches_list, batch_sizes_list)
    )

    # ------------------------------------------------------------------
    # Flush allocations from the previous benchmark before measuring
    # baseline so that the baseline is stable and representative.
    # ------------------------------------------------------------------
    gc.collect()
    if CUPY_AVAILABLE and cp_module is not None:
        try:
            cp_module.get_default_memory_pool().free_all_blocks()
            cp_module.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    gc.collect()

    # Baseline snapshots (after GC flush)
    rss_init = get_rss_mb()
    gpu_free_init, gpu_pool_init = get_gpu_snapshot()

    # ------------------------------------------------------------------
    # Graph construction + growth
    # ------------------------------------------------------------------
    initial_node_urns = cp_module.zeros((initial_nodes, num_colors), dtype=cp_module.int32)
    initial_node_urns[:, 0] = 1
    initial_node_urns[:, 1] = 1

    graph = Graph(num_nodes=initial_nodes, num_colors=num_colors,
                  use_gpu=True, node_urns=initial_node_urns)
    growth = Barabasi_Albert_Growth(graph, m=barabasi_num_connections,
                                    initial_nodes=initial_nodes)

    # Start peak sampler — it tracks the true high-water mark of RSS
    # for the entirety of this benchmark, not the process lifetime.
    sampler = PeakMemorySampler(interval=0.05)
    sampler.start()

    t_growth_start = time.time()
    try:
        for num_batches_value, batch_size_value in zip(num_batches_list, batch_sizes_list):
            run_growth_batches(num_batches=num_batches_value,
                               batch_size=batch_size_value,
                               callback=init_urns)
        csr_time_start = time.time()
        growth.finalize_growth()
        t_growth_end = time.time()
        growth_time = t_growth_end - t_growth_start
        csr_time = t_growth_end - csr_time_start
    except Exception as e:
        sampler.stop()
        print(f'Failed to grow graph: {e}')
        return None

    rss_post_growth = get_rss_mb()
    gpu_free_growth, gpu_pool_growth = get_gpu_snapshot()

    # ------------------------------------------------------------------
    # Polya process
    # ------------------------------------------------------------------
    deltas = cp_module.full((graph.num_nodes, graph.num_colors), delta_gain,
                            dtype=cp_module.int32)
    polya = Polya_Process(graph, memory_enabled=memory_enabled,
                          memory_decay_time=memory_decay_time, delta=deltas)

    t_polya_start = time.time()
    try:
        for step_i in range(polya_steps_value):
            polya.step()
            print(f'Polya step {step_i+1}/{polya_steps_value} completed. '
                  f'Current edges: {graph.num_edges}')
        t_polya_end = time.time()
        polya_time = t_polya_end - t_polya_start
    except Exception as e:
        sampler.stop()
        print(f'Failed Polya process: {e}')
        return None

    rss_final = get_rss_mb()
    gpu_free_final, gpu_pool_final = get_gpu_snapshot()

    # Stop sampler — gives us the true peak RSS during this benchmark
    peak_rss = sampler.stop()

    # ------------------------------------------------------------------
    # Compute deltas
    # ------------------------------------------------------------------
    # CPU: use the background-sampled peak minus the pre-benchmark baseline.
    # Clamped to 0 — a negative value means GC freed more than was allocated,
    # which is noise (treat as ~0 extra memory needed).
    cpu_peak_delta_mb = max(0.0, peak_rss - rss_init) if (
        peak_rss is not None and rss_init is not None) else None

    # GPU pool delta: max of growth and final snapshots minus init.
    # Taking the max of both snapshots guards against memory that was
    # freed by the time we read the final snapshot.
    if gpu_pool_init is not None:
        gpu_pool_peak = max(
            gpu_pool_growth if gpu_pool_growth is not None else gpu_pool_init,
            gpu_pool_final if gpu_pool_final is not None else gpu_pool_init,
        )
        gpu_used_pool_delta = max(0.0, gpu_pool_peak - gpu_pool_init)
    else:
        gpu_used_pool_delta = None

    result = {
        'clique_size': initial_nodes,
        'final_nodes': expected_final_nodes,
        'polya_steps': polya_steps_value,
        'num_batches': ';'.join(str(x) for x in num_batches_list),
        'batch_size': ';'.join(str(x) for x in batch_sizes_list),
        'total_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'growth_time': growth_time,
        'csr_time': csr_time,
        'polya_time': polya_time,
        'total_time': growth_time + polya_time,
        # --- memory: deltas (what this benchmark *added*) ---
        'cpu_peak_delta_mb': cpu_peak_delta_mb,   # peak RSS rise above baseline
        'gpu_used_pool_mb': gpu_used_pool_delta,  # peak GPU pool above baseline
        # --- raw snapshots for diagnostics ---
        'rss_init_mb': rss_init,
        'rss_post_growth_mb': rss_post_growth,
        'rss_final_mb': rss_final,
        'peak_rss_mb': peak_rss,
        'gpu_free_init': gpu_free_init,
        'gpu_free_growth': gpu_free_growth,
        'gpu_free_final': gpu_free_final,
        'gpu_pool_init': gpu_pool_init,
        'gpu_pool_growth': gpu_pool_growth,
        'gpu_pool_final': gpu_pool_final,
    }

    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return result


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
output_csv = 'benchmark_results.csv'

benchmark_configs = [
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32, 32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048, 32768], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32, 32, 32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048, 32768, 524288], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64, 64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048, 32768], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64, 64, 64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048, 32768, 524288], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128, 128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048, 32768], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128, 128, 128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([8, 128, 2048, 32768, 524288], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32, 32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024, 16384], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([32, 32, 32, 32, 32], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024, 16384, 262144], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64, 64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024, 16384], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([64, 64, 64, 64, 64], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024, 16384, 262144], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128, 128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024, 16384], dtype=cp_module.int32), 'polya_steps': 100},
    {'initial_nodes': 64, 'num_batches_array': cp_module.array([128, 128, 128, 128, 128], dtype=cp_module.int32),
     'batch_sizes_array': cp_module.array([4, 64, 1024, 16384, 262144], dtype=cp_module.int32), 'polya_steps': 100},
]

print("Starting benchmark scenarios...")
for cfg in benchmark_configs:
    print(f"\nRunning: initial_nodes={cfg['initial_nodes']}, "
          f"batches={cfg['num_batches_array'].tolist()}, "
          f"sizes={cfg['batch_sizes_array'].tolist()}, "
          f"polya_steps={cfg['polya_steps']}")
    result = run_benchmark(
        cfg['initial_nodes'], cfg['num_batches_array'],
        cfg['batch_sizes_array'], cfg['polya_steps'], output_csv
    )
    if result:
        print(f"  Done: final_nodes={result['final_nodes']}, "
              f"edges={result['num_edges']}, "
              f"total_time={result['total_time']:.2f}s, "
              f"cpu_peak_delta={result['cpu_peak_delta_mb']:.1f} MB, "
              f"gpu_pool_delta={result['gpu_used_pool_mb']}")
    else:
        print("  Scenario failed")

print(f"\nAll benchmarks done. Results saved to {output_csv}")