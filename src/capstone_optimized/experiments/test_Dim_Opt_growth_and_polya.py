import sys
import os
import random
import time
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

from capstone_optimized.core import Graph, Polya_Process, Switch_Network_Growth, Switched_Network_Optimized_Graph, Dim_Optimized_Switch_Network_Growth

# If user requests GPU but it's not available in this interpreter, fail fast with guidance
if not CUPY_AVAILABLE:
    print('\nGPU requested but CuPy/CUDA not available in this Python environment.')
    print('Make sure you run this script inside your virtualenv where CuPy is installed:')
    print("  . .venv\\Scripts\\Activate")
    print("  $env:PYTHONPATH = 'src'")
    print("  python src\\capstone_optimized\\experiments\\test_y.py")
    # don't exit automatically; continue so the script still runs on CPU if desired


N = int(os.environ.get('CAPSTONE_N', '175000000'))
graph = Switched_Network_Optimized_Graph(num_nodes=N, num_colors=2, use_gpu=True)

# Grow network in batches (safe mode: retries with smaller batch sizes on OOM)
growth = Dim_Optimized_Switch_Network_Growth(graph, connection_prob = 0.1)
NUM_BATCHES = int(os.environ.get('CAPSTONE_BATCHES', '1750'))
INITIAL_BATCH_SIZE = int(os.environ.get('CAPSTONE_BATCH_SIZE', '1000000'))

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

t_growth_start = time.time()
for b in range(NUM_BATCHES):
    batch_size = INITIAL_BATCH_SIZE
    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        try:
            growth.grow_batch(batch_size=batch_size)
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

            if free_mb is not None:
                print(f'Batch {b+1}/{NUM_BATCHES} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges} free_mem={free_mb:.1f}MB total={total_mb:.1f}MB')
            else:
                print(f'Batch {b+1}/{NUM_BATCHES} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges}')
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
print(f'Growth phase complete. Total growth time: {time.time() - t_growth_start:.3f}s')

print("Building CSR adjacency matrix...")
csr_time_start = time.time()
growth.finalize_growth()  # build CSR adjacency
print(f'CSR build time: {time.time() - csr_time_start:.3f}s')

# Run Polya process (timed)
print("Running Polya process...")
polya_start_time = time.time()

polya = Polya_Process(graph, delta=1)
POLYA_STEPS = int(os.environ.get('CAPSTONE_POLYA_STEPS', '100'))
for step_i in range(POLYA_STEPS):
    t0 = time.time()
    polya.step()
    t1 = time.time()
    print(f'Polya step {step_i+1}/{POLYA_STEPS} time={t1-t0:.3f}s')

print(f'Polya process complete. Total time: {time.time() - polya_start_time:.3f}s')
print('TEST_Y_DONE', 'nodes=', graph.num_nodes, 'edges=', graph.num_edges)

"""
# Optional: export urn matrix as CSV
import numpy as np
urns_np = graph.node_urns
if hasattr(urns_np, 'get'):  # if CuPy array
    urns_np = urns_np.get()
np.savetxt("node_urns.csv", urns_np, delimiter=",", fmt="%d")


# Optional: export adjacency matrix as CSV in COO format
from scipy.sparse import csr_matrix

adj = graph.adj_matrix  # could also be cupyx.scipy.sparse.csr_matrix

# If it's a CuPy CSR, convert to SciPy CSR
if 'cupyx' in str(type(adj)):
    adj = adj.get()  # cupyx -> scipy.sparse

# Save as CSV in **coordinate (COO) format**: row, col, value
coo = adj.tocoo()
np.savetxt(
    "adjacency_coo.csv",
    np.vstack((coo.row, coo.col, coo.data)).T,
    delimiter=",",
    fmt="%d"
)
"""
