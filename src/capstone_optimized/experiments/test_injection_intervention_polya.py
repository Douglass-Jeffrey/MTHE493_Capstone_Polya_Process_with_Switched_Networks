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

from capstone_optimized.core import Graph, Polya_Process, Switch_Network_Growth, Switched_Network_Optimized_Graph, Dim_Optimized_Switch_Network_Growth, Barabasi_Albert_Growth, Switch_Network_Injection_Intervention

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
                
                if free_mb is not None:
                    print(f'Batch {b+1}/{num_batches} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges} free_mem={free_mb:.1f}MB total={total_mb:.1f}MB')
                else:
                    print(f'Batch {b+1}/{num_batches} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges}')
                
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
rng = cp_module.random.default_rng()
def init_urns(b_a, bs):
    #added_node_urns = rng.integers(0, 9, size=(bs, b_a.graph.num_colors))
    added_node_urns = cp_module.zeros((bs, b_a.graph.num_colors), dtype=cp_module.int32)
    #init new urns with 8 red, 8 black
    added_node_urns[:, 0] = 1
    added_node_urns[:, 1] = 1

    b_a.graph.node_urns = cp_module.vstack([b_a.graph.node_urns, added_node_urns])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs
    return

num_colors = 2
initial_nodes = 64
num_batches = cp_module.array([256, 256, 256, ], dtype=cp_module.int32) #128
batch_sizes = cp_module.array([   8,  256, 8192, ], dtype=cp_module.int32) #65536
barabasi_num_connections = 3

num_interventions = 1
per_intervention_num_injections = 131072/2/2
intervener_urn = cp_module.array([1000,0], dtype=cp_module.int32)

polya_steps = 720
memory_enabled = True
memory_decay_time = 72
delta_gain = 1
mem_decay_loss = 1

#create initial nodes with 8 red balls, 8 black ball each
#initial_node_urns = rng.integers(0, 9, size=(initial_nodes, num_colors))
initial_node_urns = cp_module.zeros((initial_nodes, num_colors), dtype=cp_module.int32)
initial_node_urns[:, 0] = 1
initial_node_urns[:, 1] = 1

graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True, node_urns=initial_node_urns)
# Grow network in batches (safe mode: retries with smaller batch sizes on OOM)
growth = Barabasi_Albert_Growth(graph, m=barabasi_num_connections, initial_nodes=initial_nodes)

t_growth_start = time.time()
for nb, bs in zip(num_batches, batch_sizes):
    print(f'Starting growth phase: {nb} batches of size {bs}...')
    run_growth_batches(num_batches = int(nb), batch_size = int(bs), callback = init_urns)
print(f'All growth phases complete. Total growth time: {time.time() - t_growth_start:.3f}s')

t_intervention_start = time.time()
# create intervention class once we are done growing so that intervention has access to complete graph 
intervention = Switch_Network_Injection_Intervention(graph, once_per_node_iv = False)
#print(f'Starting intervention: {num_interventions} interventions with {per_intervention_num_injections} injections of urn: {intervener_urn}...')
#for i in range(num_interventions):
#    intervention.degree_centrality_optimized_intervention_step(num_injections=per_intervention_num_injections, intervener_urn=intervener_urn)
#print(f'Intervention complete. Total time: {time.time() - t_intervention_start:.3f}s')

print("Building CSR adjacency matrix...")
csr_time_start = time.time()
growth.finalize_growth()  # build CSR adjacency
print(f'CSR build time: {time.time() - csr_time_start:.3f}s')

# Run Polya process (timed)
print("Running Polya process...")
polya_start_time = time.time()

deltas = cp_module.full((graph.num_nodes, graph.num_colors), delta_gain, dtype=cp_module.int32)
mem_decay = cp_module.full((graph.num_nodes, graph.num_colors), mem_decay_loss, dtype=cp_module.int32)


polya = Polya_Process(graph, memory_enabled=memory_enabled, memory_decay_time=memory_decay_time, delta=deltas, mem_decay=mem_decay)
POLYA_STEPS = int(os.environ.get('CAPSTONE_POLYA_STEPS', f'{polya_steps}'))

num_bins = 72
hist_bins = cp_module.asnumpy(cp_module.linspace(0, 1, num_bins+1))
hist_data = cp_module.asnumpy(cp_module.zeros((polya_steps, num_bins)))
for step_i in range(POLYA_STEPS):
    t0 = time.time()
    polya.step()
    t1 = time.time()

    #collect data
    probs = ((graph.node_urns[:, 0] / cp_module.sum(graph.node_urns, axis=1)))
    hist_i, _ = cp_module.histogram(probs, hist_bins)
    hist_data[step_i] = cp_module.asnumpy(hist_i)

    if step_i == 200:
        intervention.degree_centrality_optimized_intervention_step(num_injections=per_intervention_num_injections, intervener_urn=intervener_urn)
        print(f'current num_interventions = {intervention.current_num_interventions}')

    #print info
    if step_i % 100 == 0:
        print(f'Polya step {step_i}/{POLYA_STEPS} time={t1-t0:.3f}s')

print(f'Polya process complete. Total time: {time.time() - polya_start_time:.3f}s')
print('TEST_Y_DONE', 'nodes=', graph.num_nodes, 'edges=', graph.num_edges)

row_sums = hist_data.sum(axis=1, keepdims=True)
hist_norm = hist_data/row_sums
vmax_val = float(cp_module.percentile(cp_module.array(hist_norm), 99))
vmin_val = float(cp_module.percentile(cp_module.array(hist_norm), 1))



import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

# We transpose so Time is on the X-axis and Probability is on the Y-axis
# origin='lower' ensures 0.0 probability is at the bottom
plt.imshow(hist_norm.T, aspect='auto', origin='lower', 
            extent=[0, hist_norm.shape[0], 0, 1], cmap='magma', vmin=vmin_val, vmax=vmax_val)
plt.colorbar(label='Proportion of Nodes')
plt.title("Evolution of Red Probability Distribution")
plt.xlabel("Polya Process Steps")
plt.ylabel("Probability of Red (P_i)")
plt.show()


# Optional: export urn matrix as CSV

import numpy as np
urns_np = hist_data
if hasattr(urns_np, 'get'):  # if CuPy array
    urns_np = urns_np.get()
np.savetxt("hist_data.csv", urns_np, delimiter=",", fmt="%d")

"""
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
    np.vstack((coo.row, coo.col)).T,
    delimiter=",",
    fmt="%d"
)
"""