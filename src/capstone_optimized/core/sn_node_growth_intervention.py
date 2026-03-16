from ..cupy_fallback import cp
from .graph_x import Graph

class Switch_Network_Node_Growth_Intervention:
    """
    Add new red biased nodes to the graph that will connect to the highest degree nodes.
    This is "Method 1"
    """
    def __init__(self, graph: Graph, once_per_node_iv = True):
        self.graph = graph
        self.current_num_interventions = 0
        self.once_per_node_iv = once_per_node_iv
        # information about the edge buffer and degrees of each node in the graph
        self.active_buffer = self.graph.edges[:self.graph.num_edges].flatten()
        self.degrees = cp.bincount(self.active_buffer, minlength=self.graph.num_nodes)
        self.degree_ordered_nodes = cp.flip(cp.argsort(self.degrees))
        #print(f"degree_ordered_nodes: {self.degree_ordered_nodes}")

    def degree_centrality_optimized_intervention_step(self, num_connections=1, intervener_urn=None, mid_polya_intervention=False, polya_process=None, num_interventions=1):
        """
        At each intervention step, we create num_interventions node(s) on the graph with num_connections # edges
        connecting to existing nodes. Each connected node's mega urn will thus include intervener_urn, and
        the new intervener node will have a node_urn defined by intervener_urn.

        This function will select targets for the intervener node based on degree centrality, connecting to
        the #num_connections highest degree nodes on the graph in descending order
        """
        if intervener_urn is None:
            intervener_urn = cp.ones((self.graph.num_colors,), dtype=cp.int32)

        if not hasattr(intervener_urn, "dtype"):
            intervener_urn = cp.array(intervener_urn, dtype=self.graph.node_urns.dtype)

        if intervener_urn.shape[0] != self.graph.num_colors:
            raise ValueError(
                f"Urn size mismatch. Expected {self.graph.num_colors}, got {intervener_urn.shape[0]}"
            )

        old_num_nodes = self.graph.num_nodes
        new_num_nodes = old_num_nodes + num_interventions
        self.graph.num_nodes = new_num_nodes

        #generate all new nodes at once with tiling
        new_urns = cp.tile(intervener_urn, (num_interventions, 1))
        self.graph.node_urns = cp.vstack((self.graph.node_urns, new_urns))

        # new node ids
        new_node_ids = cp.arange(old_num_nodes, new_num_nodes, dtype=cp.int32)

        # repeat each node for its connections
        src = cp.repeat(new_node_ids, num_connections)

        if self.once_per_node_iv:

            start = self.current_num_interventions
            end = start + num_interventions * num_connections

            dst = self.degree_ordered_nodes[start:end]
            
            #increment internal counter so future interventions dont act on same top nodes if once_per_node_iv is enabled
            self.current_num_interventions += num_interventions * num_connections

        #otherwise we try to target the same nodes for each intervention
        else:
            # same targets for every intervention node
            top_targets = self.degree_ordered_nodes[:num_connections]

            # repeat these targets for each new node
            dst = cp.tile(top_targets, num_interventions)

        #add new edges
        self.graph.add_edges(src, dst)
        self.graph.add_edges(dst, src)

        #resize the polya arrays and rebuild csr if we are adding new nodes in the middle of the test 
        if mid_polya_intervention:
            if polya_process is None:
                raise ValueError("Need polya_process when mid_polya_intervention=True")
            #extend the polya process's memory array howizontally to include the new nodes
            memory_extension = cp.zeros((polya_process.memory_decay_time, num_interventions), dtype=cp.int32)
            polya_process.memory_arr = cp.hstack((polya_process.memory_arr, memory_extension))

            #similarly extend the delta, and decay, these are vertical stacks of num_intervention rows
            # with columns equal to whatever the base delta, decay values in the array are
            delta_extension = cp.tile(polya_process.delta[0], (num_interventions, 1))
            polya_process.delta = cp.vstack((polya_process.delta, delta_extension))

            decay_extension = cp.tile(polya_process.mem_decay[0], (num_interventions, 1))
            polya_process.mem_decay = cp.vstack((polya_process.mem_decay, decay_extension))

            #rebuild csr when all is done in order to do next polya steps
            self.graph.build_csr()

    def random_intervention_step(self, num_connections=1, intervener_urn=None, mid_polya_intervention=False, polya_process=None, num_interventions=1):
        """
        At each intervention step, we create num_interventions node(s) on the graph with num_connections # edges
        connecting to existing nodes. Each connected node's mega urn will thus include intervener_urn, and
        the new intervener node will have a node_urn defined by intervener_urn.

        This function will select targets for the intervener node randomly
        """
        if intervener_urn is None:
            intervener_urn = cp.ones((self.graph.num_colors,), dtype=cp.int32)

        if not hasattr(intervener_urn, "dtype"):
            intervener_urn = cp.array(intervener_urn, dtype=self.graph.node_urns.dtype)

        if intervener_urn.shape[0] != self.graph.num_colors:
            raise ValueError(
                f"Urn size mismatch. Expected {self.graph.num_colors}, got {intervener_urn.shape[0]}"
            )

        old_num_nodes = self.graph.num_nodes
        new_num_nodes = old_num_nodes + num_interventions
        self.graph.num_nodes = new_num_nodes

        #generate all new nodes at once with tiling
        new_urns = cp.tile(intervener_urn, (num_interventions, 1))
        self.graph.node_urns = cp.vstack((self.graph.node_urns, new_urns))

        # new node ids
        new_node_ids = cp.arange(old_num_nodes, new_num_nodes, dtype=cp.int32)

        # repeat each node for its connections
        src = cp.repeat(new_node_ids, num_connections)
        dst = cp.random.choice(cp.arange(old_num_nodes), size=(num_interventions*num_connections), replace= (False if self.once_per_node_iv else True))

        #add new edges
        self.graph.add_edges(src, dst)
        self.graph.add_edges(dst, src)

        #resize the polya arrays and rebuild csr if we are adding new nodes in the middle of the test 
        if mid_polya_intervention:
            if polya_process is None:
                raise ValueError("Need polya_process when mid_polya_intervention=True")
            #extend the polya process's memory array howizontally to include the new nodes
            memory_extension = cp.zeros((polya_process.memory_decay_time, num_interventions), dtype=cp.int32)
            polya_process.memory_arr = cp.hstack((polya_process.memory_arr, memory_extension))

            #similarly extend the delta, and decay, these are vertical stacks of num_intervention rows
            # with columns equal to whatever the base delta, decay values in the array are
            delta_extension = cp.tile(polya_process.delta[0], (num_interventions, 1))
            polya_process.delta = cp.vstack((polya_process.delta, delta_extension))

            decay_extension = cp.tile(polya_process.mem_decay[0], (num_interventions, 1))
            polya_process.mem_decay = cp.vstack((polya_process.mem_decay, decay_extension))

            #rebuild csr when all is done in order to do next polya steps
            self.graph.build_csr()