from ..cupy_fallback import cp
from .graph_x import Graph

class Switch_Network_Injection_Intervention:
    """
    Intervene by modifying the delta values of highest degree nodes in graph
    This is "Method 3"
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        self.current_num_interventions = 0
        # information about the edge buffer and degrees of each node in the graph
        self.active_buffer = self.graph.edges[:self.graph.num_edges].flatten()
        self.degrees = cp.bincount(self.active_buffer, minlength=self.graph.num_nodes)
        self.degree_ordered_nodes = cp.flip(cp.argsort(self.degrees))
        print(f"degree_ordered_nodes: {self.degree_ordered_nodes}")

    def degree_centrality_optimized_intervention_step(self, num_injections=1, intervention_deltas=1, intervener_urn=None):
        """
        At each intervention step, select #num connections highest degree nodes from the graph.
        Each node's urn will be directly injected with the balls contained in intervener_urn

        This function will select targets for the intervener node based on degree centrality, connecting to
        the #num_injections highest degree nodes on the graph in descending order
        """
        # cast intervener_urn if necessary
        if intervener_urn is None:
            intervener_urn = cp.ones((1, self.graph.num_colors), dtype=cp.int32)
        if not hasattr(intervener_urn, 'dtype'): intervener_urn = cp.array(intervener_urn, dtype=self.graph.node_urns.dtype)

        if intervener_urn.shape[0] != self.graph.num_colors:
            raise ValueError(f"Urn size mismatch. Expected {self.graph.num_colors}, got {intervener_urn.shape[0]}")

        #add intervener node contents to num_injections next highest degree nodes
        dst = self.degree_ordered_nodes[self.current_num_interventions:self.current_num_interventions + num_injections]
        self.graph.node_urns[dst] += intervener_urn
        # add num injections to num interventions so that we inject on subsequent highest nodes on future injections
        self.current_num_interventions += num_injections
        print(f" injected intervention urn: {intervener_urn} into nodes: {dst}")
        

