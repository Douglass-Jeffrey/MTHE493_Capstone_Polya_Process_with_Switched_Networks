import sys
import os
import random
import time
import csv
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available, CPU memory tracking disabled")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Ensure we know whether CuPy is available in the current interpreter
try:
    from capstone_optimized import cupy_fallback as cf
    cp_module = getattr(cf, 'cp', None)
    CUPY_AVAILABLE = getattr(cf, 'CUPY_AVAILABLE', False)
    print('cupy_fallback: cp module =', getattr(cp_module, '__name__', None), 'CUPY_AVAILABLE=', CUPY_AVAILABLE)
except Exception as _e:
    print('Could not import capstone_optimized.cupy_fallback:', _e)
    CUPY_AVAILABLE = False

from capstone_optimized.core import Graph, Polya_Process, Switch_Network_Growth, Switched_Network_Optimized_Graph, Dim_Optimized_Switch_Network_Growth, Barabasi_Albert_Growth, Switch_Network_Node_Growth_Intervention

# If user requests GPU but it's not available in this interpreter, fail fast with guidance
if not CUPY_AVAILABLE:
    print('\nGPU requested but CuPy/CUDA not available in this Python environment.')
    print('Make sure you run this script inside your virtualenv where CuPy is installed:')
    print("  . .venv\\Scripts\\Activate")
    print("  $env:PYTHONPATH = 'src'")
    print("  python src\\capstone_optimized\\experiments\\test_y.py")
    # don't exit automatically; continue so the script still runs on CPU if desired

def _is_oom_exception(exc):
    msg = str(exc).lower()
    if isinstance(exc, MemoryError):
        return True
    if 'out of memory' in msg:
        return True
    if 'no memory' in msg:
        return True
    if 'cuda' in msg and 'memory' in msg:
        return True
    return False

def run_growth_batches(num_batches, batch_size, callback = None):
    for b in range(num_batches):
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                growth.grow_batch(batch_size=batch_size, urn_addition_callback=callback)
                t1 = time.time()
                # report GPU memory if available
                free_mb = total_mb = None
                if graph.use_gpu and CUPY_AVAILABLE and cp_module is not None:
                    try:
                        free, total = cp_module.cuda.runtime.memGetInfo()
                        free_mb = free / 1024 ** 2
                        total_mb = total / 1024 ** 2
                    except Exception:
                        free_mb = total_mb = None
                
                #if free_mb is not None:
                    #print(f'Batch {b+1}/{num_batches} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges} free_mem={free_mb:.1f}MB total={total_mb:.1f}MB')
                #else:
                    #print(f'Batch {b+1}/{num_batches} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges}')
                
                break
            except Exception as e:
                if _is_oom_exception(e):
                    # reduce batch size and retry
                    if batch_size <= 1:
                        print('Batch size reduced to 1 but still OOM; aborting')
                        raise
                    new_bs = max(1, batch_size // 2)
                    print(f'Encountered OOM on batch (size={batch_size}). Reducing to {new_bs} and retrying (attempt {attempt}).')
                    batch_size = new_bs
                    continue
                else:
                    # re-raise unexpected exceptions
                    raise

def init_urns(b_a, bs):
    added_node_urns = cp_module.zeros((bs, b_a.graph.num_colors), dtype=cp_module.int32)
    #init new urns with 10 red, 10 black
    added_node_urns[:, 0] = 1
    added_node_urns[:, 1] = 1

    b_a.graph.node_urns = cp_module.vstack([b_a.graph.node_urns, added_node_urns])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs
    return

def get_memory_usage():
    gpu_free = gpu_total = gpu_used_pool = None
    if CUPY_AVAILABLE and cp_module is not None:
        try:
            free, total = cp_module.cuda.runtime.memGetInfo()
            gpu_free = free / 1024 ** 2
            gpu_total = total / 1024 ** 2
        except Exception:
            pass

        try:
            # Attempt CuPy memory pool usage (more accurate for allocated tensors)
            mempool = cp_module.get_default_memory_pool()
            gpu_used_pool = mempool.used_bytes() / 1024 ** 2
        except Exception:
            gpu_used_pool = None

    cpu_mem = None
    cpu_mem_peak = None
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            info = process.memory_info()
            cpu_mem = info.rss / 1024 ** 2
            cpu_mem_peak = getattr(info, 'peak_wset', None)
            if cpu_mem_peak is not None:
                cpu_mem_peak = cpu_mem_peak / 1024 ** 2
        except Exception:
            try:
                cpu_mem = psutil.virtual_memory().used / 1024 ** 2
            except Exception:
                cpu_mem = None
    return gpu_free, gpu_total, gpu_used_pool, cpu_mem, cpu_mem_peak

def run_benchmark(initial_nodes, num_batches_array, batch_sizes_array, polya_steps_value, output_csv):
    global graph, growth  # to access in run_growth_batches
    
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

    expected_final_nodes = initial_nodes + sum(n * b for n, b in zip(num_batches_list, batch_sizes_list))

    # create initial nodes with 1 red ball, 1 black ball each
    initial_node_urns = cp_module.zeros((initial_nodes, num_colors), dtype=cp_module.int32)
    initial_node_urns[:, 0] = 1
    initial_node_urns[:, 1] = 1

    graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True, node_urns=initial_node_urns)
    growth = Barabasi_Albert_Growth(graph, m=barabasi_num_connections, initial_nodes=initial_nodes)

    gpu_free_init, gpu_total_init, gpu_used_pool_init, cpu_mem_init, cpu_mem_peak_init = get_memory_usage()

    t_growth_start = time.time()
    try:
        for num_batches_value, batch_size_value in zip(num_batches_list, batch_sizes_list):
            run_growth_batches(num_batches=num_batches_value, batch_size=batch_size_value, callback=init_urns)
        csr_time_start = time.time()
        growth.finalize_growth()  # build CSR adjacency
        t_growth_end = time.time()
        growth_time = t_growth_end - t_growth_start
        csr_time = t_growth_end - csr_time_start
    except Exception as e:
        print(f'Failed to grow graph with batch phases {list(zip(num_batches_list, batch_sizes_list))}: {e}')
        return None

    gpu_free_growth, gpu_total_growth, gpu_used_pool_growth, cpu_mem_growth, cpu_mem_peak_growth = get_memory_usage()

    deltas = cp_module.full((graph.num_nodes, graph.num_colors), delta_gain, dtype=cp_module.int32)
    polya = Polya_Process(graph, memory_enabled=memory_enabled, memory_decay_time=memory_decay_time, delta=deltas)

    t_polya_start = time.time()
    try:
        for step_i in range(polya_steps_value):
            polya.step()
            print(f'Polya step {step_i+1}/{polya_steps_value} completed. Current edges: {graph.num_edges}')
        t_polya_end = time.time()
        polya_time = t_polya_end - t_polya_start
    except Exception as e:
        print(f'Failed Polya process: {e}')
        return None

    gpu_free_final, gpu_total_final, gpu_used_pool_final, cpu_mem_final, cpu_mem_peak_final = get_memory_usage()

    gpu_used_meminfo = (gpu_total_init - gpu_free_growth) if gpu_free_growth is not None and gpu_total_init is not None else None
    gpu_used_pool = (gpu_used_pool_final - gpu_used_pool_init) if gpu_used_pool_final is not None and gpu_used_pool_init is not None else None
    cpu_used = (cpu_mem_final - cpu_mem_init) if cpu_mem_final is not None and cpu_mem_init is not None else None
    cpu_peak_used = cpu_mem_peak_final if cpu_mem_peak_final is not None else cpu_mem_final


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
        'gpu_used_meminfo_mb': gpu_used_meminfo,
        'gpu_used_pool_mb': gpu_used_pool,
        'cpu_used_mb': cpu_used,
        'cpu_peak_mb': cpu_peak_used,
        'gpu_free_init': gpu_free_init,
        'gpu_free_growth': gpu_free_growth,
        'gpu_free_final': gpu_free_final,
        'cpu_mem_init': cpu_mem_init,
        'cpu_mem_growth': cpu_mem_growth,
        'cpu_mem_final': cpu_mem_final,
        'cpu_mem_peak_init': cpu_mem_peak_init,
        'cpu_mem_peak_growth': cpu_mem_peak_growth,
        'cpu_mem_peak_final': cpu_mem_peak_final
    }

    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(result)

    return result

# Benchmarking parameters
output_csv = 'benchmark_results.csv'

# Define multiple benchmark scenarios with explicit batch phase control
benchmark_configs = [
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32, 32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048, 32768], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32, 32, 32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048, 32768, 524288], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64, 64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048, 32768], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64, 64, 64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048, 32768, 524288], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128, 128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048, 32768], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128, 128, 128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([8, 128, 2048, 32768, 524288], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32, 32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024, 16384], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([32, 32, 32, 32, 32], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024, 16384, 262144], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64, 64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024, 16384], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([64, 64, 64, 64, 64], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024, 16384, 262144], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128, 128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024, 16384], dtype=cp_module.int32),
        'polya_steps': 100
    },
    {
        'initial_nodes': 64,
        'num_batches_array': cp_module.array([128, 128, 128, 128, 128], dtype=cp_module.int32),
        'batch_sizes_array': cp_module.array([4, 64, 1024, 16384, 262144], dtype=cp_module.int32),
        'polya_steps': 100
    },
]

print("Starting multiple benchmark scenarios...")
for cfg in benchmark_configs:
    print(f"Running scenario: initial_nodes={cfg['initial_nodes']}, batches={cfg['num_batches_array'].tolist()}, sizes={cfg['batch_sizes_array'].tolist()}, polya_steps={cfg['polya_steps']}...")
    result = run_benchmark(cfg['initial_nodes'], cfg['num_batches_array'], cfg['batch_sizes_array'], cfg['polya_steps'], output_csv)
    if result:
        print(f"Completed: final_nodes={result['final_nodes']}, edges={result['num_edges']}, total_time={result['total_time']:.2f}s")
    else:
        print("Scenario failed")

print("All benchmarks completed. Results saved to", output_csv)
