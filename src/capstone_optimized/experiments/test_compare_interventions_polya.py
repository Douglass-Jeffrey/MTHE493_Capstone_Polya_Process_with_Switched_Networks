import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from capstone_optimized import cupy_fallback as cf
    cp_module = getattr(cf, 'cp', None)
    CUPY_AVAILABLE = getattr(cf, 'CUPY_AVAILABLE', False)
    print('cupy_fallback: cp module =', getattr(cp_module, '__name__', None), 'CUPY_AVAILABLE=', CUPY_AVAILABLE)
except Exception as _e:
    print('Could not import capstone_optimized.cupy_fallback:', _e)
    cp_module = None
    CUPY_AVAILABLE = False

from capstone_optimized.core import Graph, Polya_Process, Barabasi_Albert_Growth, Switch_Network_Injection_Intervention
from capstone_optimized.core.sn_node_growth_intervention import Switch_Network_Node_Growth_Intervention


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


def init_urns(b_a, bs):
    added_node_urns = cp_module.zeros((bs, b_a.graph.num_colors), dtype=cp_module.int32)
    added_node_urns[:, 0] = 10
    added_node_urns[:, 1] = 10
    b_a.graph.node_urns = cp_module.vstack([b_a.graph.node_urns, added_node_urns])
    b_a.graph.num_nodes = b_a.graph.num_nodes + bs
    return


def run_growth(graph, m, initial_nodes, num_batches_array, batch_sizes_array):
    growth = Barabasi_Albert_Growth(graph, m=m, initial_nodes=initial_nodes)
    for nb, bs in zip(num_batches_array, batch_sizes_array):
        nb_i = int(nb)
        bs_i = int(bs)
        for b in range(nb_i):
            attempt = 0
            while True:
                attempt += 1
                try:
                    growth.grow_batch(batch_size=bs_i, urn_addition_callback=init_urns)
                    break
                except Exception as e:
                    if _is_oom_exception(e):
                        if bs_i <= 1:
                            raise
                        bs_i = max(1, bs_i // 2)
                        print(f'OOM during growth; reducing batch_size to {bs_i} and retrying')
                        continue
                    else:
                        raise
    # Note: finalize_growth() should be called after all growth and interventions are complete
    return growth


def compare_interventions():
    # parameter grids (low, medium, high)
    m_values = [1, 3, 5]
    initial_nodes_values = [32, 128, 512]
    initial_node_urns_values = [1, 10, 100, 1000]
    memory_opts = [False, True]
    delta_values = [5, 10, 20]  # low, medium, high
    mem_decay_values = [1, 3, 5]  # low, medium, high

    # use same sizes/scope as test_node_growth_intervention_polya.py
    num_batches = cp_module.array([1024, 512, 256, 16], dtype=cp_module.int32)
    batch_sizes = cp_module.array([8, 128, 2048, 65536], dtype=cp_module.int32)

    num_interventions = 256
    per_intervention_num_connections = 1024
    per_intervention_num_injections = 1024
    intervener_urn = cp_module.array([1024, 0], dtype=cp_module.int32)
    polya_steps = 1024

    results = []

    for m in m_values:
        for initial_nodes in initial_nodes_values:
            for initial_urn in initial_node_urns_values:    
                for memory_enabled in memory_opts:
                    # if memory enabled, iterate over delta and mem_decay values
                    delta_opts = delta_values if memory_enabled else [None]
                    mem_decay_opts = mem_decay_values if memory_enabled else [None]
                    
                    for delta_val in delta_opts:
                        for mem_decay_val in mem_decay_opts:
                            print(f"\n{'='*100}")
                            print(f"TEST: m={m}, initial_nodes={initial_nodes}, initial_urn={initial_urn}, memory={memory_enabled}, delta={delta_val}, mem_decay={mem_decay_val}")
                            print(f"{'='*100}")
                            
                            # build base initial urns
                            num_colors = 2
                            initial_node_urns = cp_module.zeros((initial_nodes, num_colors), dtype=cp_module.int32)
                            initial_node_urns[:, 0] = initial_urn
                            initial_node_urns[:, 1] = initial_urn

                            # ===== INJECTION INTERVENTION TEST =====
                            print("\n[INJECTION] Phase 1: Growing network...")
                            graph_inj = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=CUPY_AVAILABLE, node_urns=initial_node_urns.copy())
                            inj_growth = run_growth(graph_inj, m=m, initial_nodes=initial_nodes, num_batches_array=num_batches, batch_sizes_array=batch_sizes)
                            
                            print(f"[INJECTION] Phase 2: Running {num_interventions} injection interventions...")
                            injection = Switch_Network_Injection_Intervention(graph_inj)
                            try:
                                for _i in range(num_interventions):
                                    injection.degree_centrality_optimized_intervention_step(num_injections=per_intervention_num_injections, intervener_urn=intervener_urn)
                            except Exception as e:
                                print('Injection intervention error:', e)

                            # Finalize CSR after growth and interventions
                            print("[INJECTION] Finalizing CSR adjacency matrix...")
                            inj_growth.finalize_growth()

                            # Create Polya process with appropriate parameters
                            print(f"[INJECTION] Phase 3: Running {polya_steps} Polya steps...")
                            if memory_enabled:
                                deltas = cp_module.full((graph_inj.num_nodes, graph_inj.num_colors), delta_val, dtype=cp_module.int32)
                                mem_decay = cp_module.full((graph_inj.num_nodes, graph_inj.num_colors), mem_decay_val, dtype=cp_module.int32)
                                polya_inj = Polya_Process(graph_inj, delta=deltas, memory_enabled=True, memory_decay_time=256, mem_decay=mem_decay)
                            else:
                                polya_inj = Polya_Process(graph_inj, memory_enabled=False, memory_decay_time=256)

                            # Track proportions for injection
                            proportions_inj = []
                            for step_i in range(polya_steps):
                                polya_inj.step()
                                mean_red = cp_module.asnumpy(cp_module.mean(graph_inj.node_urns[:, 0] / cp_module.sum(graph_inj.node_urns, axis=1)))
                                mean_black = cp_module.asnumpy(cp_module.mean(graph_inj.node_urns[:, 1] / cp_module.sum(graph_inj.node_urns, axis=1)))
                                proportions_inj.append([mean_red, mean_black])
                                if (step_i + 1) % 200 == 0:
                                    print(f"  [INJECTION] Polya step {step_i+1}/{polya_steps}")

                            mean_red_inj = proportions_inj[-1][0]  # final proportion
                            print(f"[INJECTION] Complete. Final red proportion: {mean_red_inj:.4f}")

                            # ===== NODE-GROWTH INTERVENTION TEST =====
                            print("\n[NODE-GROWTH] Phase 1: Growing network...")
                            graph_ng = Graph(num_nodes=initial_nodes, num_colors=num_colors, use_gpu=CUPY_AVAILABLE, node_urns=initial_node_urns.copy())
                            ng_growth = run_growth(graph_ng, m=m, initial_nodes=initial_nodes, num_batches_array=num_batches, batch_sizes_array=batch_sizes)
                            
                            print(f"[NODE-GROWTH] Phase 2: Running {num_interventions} node-growth interventions...")
                            node_growth = Switch_Network_Node_Growth_Intervention(graph_ng)
                            try:
                                for _i in range(num_interventions):
                                    node_growth.degree_centrality_optimized_intervention_step(num_connections=per_intervention_num_connections, intervener_urn=intervener_urn)
                            except Exception as e:
                                print('Node-growth intervention error:', e)

                            # Finalize CSR after growth and interventions (new nodes were added)
                            print("[NODE-GROWTH] Finalizing CSR adjacency matrix...")
                            ng_growth.finalize_growth()

                            # Create Polya process with appropriate parameters
                            print(f"[NODE-GROWTH] Phase 3: Running {polya_steps} Polya steps...")
                            if memory_enabled:
                                deltas = cp_module.full((graph_ng.num_nodes, graph_ng.num_colors), delta_val, dtype=cp_module.int32)
                                mem_decay = cp_module.full((graph_ng.num_nodes, graph_ng.num_colors), mem_decay_val, dtype=cp_module.int32)
                                polya_ng = Polya_Process(graph_ng, delta=deltas, memory_enabled=True, memory_decay_time=256, mem_decay=mem_decay)
                            else:
                                polya_ng = Polya_Process(graph_ng, memory_enabled=False, memory_decay_time=256)

                            # Track proportions for node-growth
                            proportions_ng = []
                            for step_i in range(polya_steps):
                                polya_ng.step()
                                mean_red = cp_module.asnumpy(cp_module.mean(graph_ng.node_urns[:, 0] / cp_module.sum(graph_ng.node_urns, axis=1)))
                                mean_black = cp_module.asnumpy(cp_module.mean(graph_ng.node_urns[:, 1] / cp_module.sum(graph_ng.node_urns, axis=1)))
                                proportions_ng.append([mean_red, mean_black])
                                if (step_i + 1) % 200 == 0:
                                    print(f"  [NODE-GROWTH] Polya step {step_i+1}/{polya_steps}")

                            mean_red_ng = proportions_ng[-1][0]  # final proportion
                            print(f"[NODE-GROWTH] Complete. Final red proportion: {mean_red_ng:.4f}")

                            # Save proportions to files
                            delta_str = f"delta{delta_val}" if memory_enabled else "delta_NA"
                            decay_str = f"decay{mem_decay_val}" if memory_enabled else "decay_NA"
                            
                            inj_file = f"proportions_inj_m{m}_init{initial_nodes}_urn{initial_urn}_mem{memory_enabled}_{delta_str}_{decay_str}.csv"
                            ng_file = f"proportions_ng_m{m}_init{initial_nodes}_urn{initial_urn}_mem{memory_enabled}_{delta_str}_{decay_str}.csv"
                            
                            # Convert to numpy arrays
                            prop_inj_np = np.array(proportions_inj)
                            prop_ng_np = np.array(proportions_ng)
                            
                            np.savetxt(inj_file, prop_inj_np, delimiter=",", fmt="%.6f", header="red_prop,black_prop", comments="")
                            np.savetxt(ng_file, prop_ng_np, delimiter=",", fmt="%.6f", header="red_prop,black_prop", comments="")

                            results.append({
                                'm': m,
                                'initial_nodes': initial_nodes,
                                'initial_urn': initial_urn,
                                'memory': memory_enabled,
                                'delta': delta_val if memory_enabled else None,
                                'mem_decay': mem_decay_val if memory_enabled else None,
                                'mean_red_injection': float(mean_red_inj),
                                'mean_red_node_growth': float(mean_red_ng)
                            })

                            print(f"m={m} init={initial_nodes} urn={initial_urn} mem={memory_enabled} delta={delta_val} decay={mem_decay_val} -> injection={mean_red_inj:.4f} ng={mean_red_ng:.4f}")

    print('\nSummary Results:')
    print("="*120)
    for r in results:
        print(f"m={r['m']:<3} init={r['initial_nodes']:<4} urn={r['initial_urn']:<5} mem={str(r['memory']):<5} delta={str(r['delta']):<6} decay={str(r['mem_decay']):<6} inj_red={r['mean_red_injection']:.4f} ng_red={r['mean_red_node_growth']:.4f}")
    print("="*120)


if __name__ == '__main__':
    if cp_module is None:
        print('No array backend available (cp_module is None). Ensure capstone_optimized.cupy_fallback provides cp or run in an environment with NumPy/CuPy configured.')
    else:
        compare_interventions()
