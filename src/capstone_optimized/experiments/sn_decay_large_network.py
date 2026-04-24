import sys
import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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


def init_urns(b_a, bs):
    added_node_urns = cp_module.zeros((bs, b_a.graph.num_colors), dtype=cp_module.int32)
    added_node_urns[:, 1] = 1
    added_node_urns[:, 0] = 1


    b_a.graph.node_urns = cp_module.vstack([b_a.graph.node_urns, added_node_urns])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs
    return

COST_1000_CONNECTIONS = 8 #cpm value for instagram
POLY_STEPS = 720 # number of hours in a month
awarness = np.zeros(POLY_STEPS)

for decay in [150, 188]:
    for bb_connec in [1, 3, 5]:
        #------------------------Variables for model set up----------------------------    
        awarness = np.zeros(int(POLY_STEPS/10)) # array to track awarnesss percentage over simulation steps
        cost = 0 #varibable to track coost of ech scenario
        num_colors = 2
        initial_nodes = 8
        #Below results in 149000 node graph to start simulation with
        num_batches = cp_module.array([256, 256, 256, 128, ], dtype=cp_module.int32) #128
        batch_sizes = cp_module.array([8, 256, 8192, 65536, ], dtype=cp_module.int32) #65536
        intervention_deltas=1
        #create initial nodes with no red balls, 1 black ball each
        initial_node_urns = cp_module.zeros((initial_nodes, num_colors), dtype=cp_module.int32)
        initial_node_urns[:, 1] = 1
        initial_node_urns[:, 0] = 1

        #--------------------Variable set up for each scenaario------------------------
        barabasi_num_connections = bb_connec
        intervener_urn = cp_module.array([1000,0], dtype=cp_module.int32)
        per_intervention_num_connections = 1000
        #intervention_prob = float(entrie["intervention_prob"])
        #title = "test_" + str(entrie["test"])
        num_interventions = 4*5
        num_trials = 1

        
        #-------------------------Build and grow BB graph------------------------------
        graph = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=True, node_urns=initial_node_urns)
        # Grow network in batches (safe mode: retries with smaller batch sizes on OOM)
        growth = Barabasi_Albert_Growth(graph, m=barabasi_num_connections, initial_nodes=initial_nodes)

        t_growth_start = time.time()
        for nb, bs in zip(num_batches, batch_sizes):
            print(f'Starting growth phase: {nb} batches of size {bs}...')
            run_growth_batches(num_batches = int(nb), batch_size = int(bs), callback = init_urns)
        print(f'All growth phases complete. Total growth time: {time.time() - t_growth_start:.3f}s')

        #-------Run Polya Process with Switched Network Growth and Interventions-------
        intervention = Switch_Network_Node_Growth_Intervention(graph, once_per_node_iv=True)
        print("Running Polya process...")
        polya_start_time = time.time()

        memory_decay_time = 72
        delta_gain = 1
        mem_decay_loss = 1
        deltas = cp_module.full((graph.num_nodes, graph.num_colors), delta_gain, dtype=cp_module.int32)
        mem_decay = cp_module.full((graph.num_nodes, graph.num_colors), mem_decay_loss, dtype=cp_module.int32)
        polya = Polya_Process(graph, memory_enabled=True, memory_decay_time=1*memory_decay_time, delta=deltas, mem_decay=mem_decay )
        #awarness[0] = np.sum((graph.node_urns[:, 0] / (graph.node_urns[:,0] + graph.node_urns[:,1]) ) > 0.60) / graph.node_urns.shape[0]
        num_bins = 80
        hist_bins = cp_module.array(cp_module.linspace(0, 1, num_bins+1))
        hist_data = cp_module.array(cp_module.zeros((POLY_STEPS, num_bins)))
        for step_i in range(POLY_STEPS):
            t0 = time.time()
            polya.step()
            # Randomly decide whether to add an intervention this step or add a new user
            if (random.random() < math.exp(-1 * step_i / decay)):
                intervention.degree_centrality_optimized_intervention_step(num_connections=per_intervention_num_connections, 
                                                                        intervener_urn=intervener_urn, 
                                                                        mid_polya_intervention=True, 
                                                                        polya_process=polya,
                                                                        num_interventions=num_interventions
                                                                        )
                #cost of intervention is cost of connections plus cost of red balls in intervener urn
                cost += num_interventions*(COST_1000_CONNECTIONS) 
            #else:
            # run_growth_batches(num_batches = 1, batch_size = 1, callback = init_urns)
            
            #track data for each step 
            probs = ((graph.node_urns[:, 0] / cp_module.sum(graph.node_urns, axis=1)))
            hist_i, _ = cp_module.histogram(probs, hist_bins)
            hist_data[step_i] = cp_module.array(hist_i)

            t1 = time.time()
            print(f'Polya step {step_i+1}/{POLY_STEPS} time={t1-t0:.3f}s')

            #print(f'Polya process complete. Total time: {time.time() - polya_start_time:.3f}s')
            #print('TEST_Y_DONE', 'nodes=', graph.num_nodes, 'edges=', graph.num_edges)

        #----------------------------------Save Data--------------------------------
        np.savetxt(f"test_sn_decay_{decay}bb_{bb_connec}.csv", hist_data, delimiter=",", fmt='%d')
