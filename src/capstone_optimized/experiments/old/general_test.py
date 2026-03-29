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

from capstone_optimized.core import Graph, Polya_Process, Barabasi_Albert_Growth, Switch_Network_Injection_Intervention, Switch_Network_Node_Growth_Intervention


# If user requests GPU but it's not available in this interpreter, fail fast with guidance
if not CUPY_AVAILABLE:
    print('\nGPU requested but CuPy/CUDA not available in this Python environment.')
    print('Make sure you run this script inside your virtualenv where CuPy is installed:')
    print("  . .venv\\Scripts\\Activate")
    print("  $env:PYTHONPATH = 'src'")
    print("  python src\\capstone_optimized\\experiments\\test_y.py")
    # don't exit automatically; continue so the script still runs on CPU if desired

def report_mem():
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

def run_growth_batches(growth, graph, num_batches, batch_size, callback = None):
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
                
                if ((free_mb is not None) and (b%100 == 0)):
                    print(f'Batch {b}/{num_batches} size={batch_size} time={t1-t0:.3f}s edges={graph.num_edges} free_mem={free_mb:.1f}MB total={total_mb:.1f}MB')
                
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
    #added_node_urns = rng.integers(0, 9, size=(bs, b_a.graph.num_colors))
    added_node_urns = cp_module.zeros((bs, b_a.graph.num_colors), dtype=cp_module.int32)
    #init new urns with 8 red, 8 black
    added_node_urns[:, 0] = 1
    added_node_urns[:, 1] = 1

    b_a.graph.node_urns = cp_module.vstack([b_a.graph.node_urns, added_node_urns])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs
    return

def main():
    rng = cp_module.random.default_rng()

    #NODE FEATURES
    num_colors = 2
    
    #GRAPH AND GROWTH FEATURES
    initial_nodes = 64
    num_batches = cp_module.array([256, 256, 256, 128], dtype=cp_module.int32) #128
    batch_sizes = cp_module.array([   8,  256, 8192, 65536], dtype=cp_module.int32) #65536
    barabasi_num_connections = 3
    #create initial nodes with 8 red balls, 8 black ball each
    #initial_node_urns = rng.integers(0, 9, size=(initial_nodes, num_colors))
    initial_node_urns = cp_module.zeros((initial_nodes, num_colors), dtype=cp_module.int32)
    initial_node_urns[:, 0] = 1
    initial_node_urns[:, 1] = 1

    #INTERVENTION FEATURES
    num_interventions = 1
    num_injections = 131072/2/2
    num_connections = None
    intervener_urn = cp_module.array([1000,0], dtype=cp_module.int32)
    once_per_node_iv = False
    intervention_steps = 144
    intervention_class = Switch_Network_Injection_Intervention
    intervention_func = intervention_class.degree_centrality_optimized_intervention_step

    #POLYA FEATURES
    polya_steps = 720+144
    memory_enabled = True
    memory_decay_time = 72
    delta_gain = 1
    mem_decay_loss = 1

    #HISTOGRAM INFORMATION
    num_bins = 50
    hist_bins = cp_module.asnumpy(cp_module.linspace(0, 1, num_bins+1))
    hist_data = cp_module.asnumpy(cp_module.zeros((polya_steps, num_bins)))

    ####################################################################################################
    print(f"Begin test\n")
    print(f"Init Graph with parameters: initial_nodes={initial_nodes},\n"
            f"num_batches={num_batches},\n"
            f"batch_sizes={batch_sizes},\n"
            f"barabasi_num_connections={barabasi_num_connections},\n"
            f"initial_node_urns={initial_node_urns}\n")
    graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True, node_urns=initial_node_urns)
    growth = Barabasi_Albert_Growth(graph, m=barabasi_num_connections, initial_nodes=initial_nodes)

    t_growth_start = time.time()
    for nb, bs in zip(num_batches, batch_sizes):
        print(f'Starting growth phase: {nb} batches of size {bs}...')
        run_growth_batches(growth=growth, graph=graph, num_batches = int(nb), batch_size = int(bs), callback = init_urns)
    growth.finalize_growth()  # build CSR adjacency
    print(f'All growth phases complete. Total growth time: {time.time() - t_growth_start:.3f}s')
    ####################################################################################################
    print(f"Init Intervention with parameters: num_interventions={num_interventions},\n"
            f"num_injections={num_injections},\n"
            f"num_connections={num_connections},\n"
            f"intervener_urn={intervener_urn},\n"
            f"intervention_class={intervention_class},\n"
            f"intervention_func={intervention_func},\n"
            f"once_per_node_iv={once_per_node_iv},\n"
            f"intervention_steps={intervention_steps}\n")
    t_intervention_start = time.time()
    intervention = intervention_class(graph = graph, once_per_node_iv = False)
    ####################################################################################################

    #Define deltas and mem decay
    deltas = cp_module.full((graph.num_nodes, graph.num_colors), delta_gain, dtype=cp_module.int32)
    mem_decay = cp_module.full((graph.num_nodes, graph.num_colors), mem_decay_loss, dtype=cp_module.int32)

    print(f"Init Polya Process with parameters: polya_steps={polya_steps},\n"
            f"memory_enabled={memory_enabled},\n"
            f"memory_decay_time={memory_decay_time},\n"
            f"delta_gain={delta_gain},\n"
            f"mem_decay={mem_decay_loss},\n"
            f"deltas={deltas},\n"
            f"mem_decay={mem_decay}\n")
    polya_start_time = time.time()
    polya = Polya_Process(graph, memory_enabled=memory_enabled, memory_decay_time=memory_decay_time, delta=deltas, mem_decay=mem_decay)

    for step_i in range(polya_steps):
        t0 = time.time()
        polya.step()
        t1 = time.time()

        if step_i % 100 == 0:
            print(f'Polya step {step_i}/{polya_steps} time={t1-t0:.3f}s')

        #collect data
        probs = ((graph.node_urns[:, 0] / cp_module.sum(graph.node_urns, axis=1)))
        hist_i, _ = cp_module.histogram(probs, hist_bins)
        hist_data[step_i] = cp_module.asnumpy(hist_i)

        if step_i == intervention_steps:

            if intervention_class is Switch_Network_Injection_Intervention:
                intervention_func(self=intervention,
                                    num_injections=num_injections,
                                    intervener_urn=intervener_urn)

            if intervention_class is Switch_Network_Node_Growth_Intervention:
                intervention_func(self=intervention,
                                    num_connections=num_connections,
                                    intervener_urn=intervener_urn,
                                    mid_polya_intervention=True,
                                    polya_process=polya,
                                    num_interventions = num_interventions)

    print(f'Polya process complete. Total time: {time.time() - polya_start_time:.3f}s')
    print('TEST_Y_DONE', 'nodes=', graph.num_nodes, 'edges=', graph.num_edges)

    # Optional: export urn matrix as CSV
    import numpy as np
    urns_np = hist_data
    if hasattr(urns_np, 'get'):  # if CuPy array
        urns_np = urns_np.get()
    np.savetxt("hist_data_i.csv", urns_np, delimiter=",", fmt="%d")

if __name__ == "__main__":
    main()
