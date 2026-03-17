from ..cupy_fallback import cp
from .graph_x import Graph

class Switch_Network_Injection_Intervention:
    """
    Inject balls directly highest degree graph nodes
    This is "Method 2"
    """
    def __init__(self, graph: Graph, once_per_node_iv = True):
        self.graph = graph
        self.current_num_interventions = 0
        self.once_per_node_iv = once_per_node_iv
        # information about the edge buffer and degrees of each node in the graph
        self.active_buffer = self.graph.edges[:self.graph.num_edges].flatten()
        self.degrees = cp.bincount(self.active_buffer, minlength=self.graph.num_nodes)
        self.degree_ordered_nodes = cp.flip(cp.argsort(self.degrees))
        
        # Placeholder for Eigenvector logic (computed only when needed)
        self.eigen_ordered_nodes = None

    def _compute_eigen_order(self, max_iter=100, tol=1e-6):
        """Do power iteration to calculate eigenvector centrality"""
        if self.graph.adj_matrix is None:
            self.graph.build_csr()

        # Power Iteration
        x = cp.ones(self.graph.num_nodes, dtype=cp.float32)
        for _ in range(max_iter):
            x_next = self.graph.adj_matrix.dot(x)
            norm = cp.linalg.norm(x_next)
            if norm == 0: break
            x_next /= norm
            if cp.linalg.norm(x_next - x) < tol:
                x = x_next
                break
            x = x_next
            
        return cp.flip(cp.argsort(x))

    def eigenvector_centrality_optimized_intervention_step(self, num_injections=1, intervener_urn=None):
        """
        Intervention Method 3: Targets nodes based on Eigenvector Centrality.
        Higher scores go to nodes connected to other highly-connected nodes.
        """
        if self.eigen_ordered_nodes is None:
            self.eigen_ordered_nodes = self._compute_eigen_order()

        if intervener_urn is None:
            intervener_urn = cp.ones((self.graph.num_colors,), dtype=cp.int32)
        if not hasattr(intervener_urn, 'dtype'): 
            intervener_urn = cp.array(intervener_urn, dtype=self.graph.node_urns.dtype)

        #get targets from the eigenvector-sorted list
        start = self.current_num_interventions
        end = start + num_injections
        dst = self.eigen_ordered_nodes[start:end]
        
        self.graph.node_urns[dst] += intervener_urn

        if self.once_per_node_iv:
            self.current_num_interventions += num_injections

    def degree_centrality_optimized_intervention_step(self, num_injections=1, intervener_urn=None):
        """
        At each intervention step, select #num connections highest degree nodes from the graph.
        Each node's urn will be directly injected with the balls contained in intervener_urn

        This function will select targets for the intervener node based on degree centrality, connecting to
        the #num_injections highest degree nodes on the graph in descending order
        """
        # cast intervener_urn if necessary
        if intervener_urn is None:
            intervener_urn = cp.ones((1, self.graph.num_colors), dtype=cp.int32)
        if not hasattr(intervener_urn, 'dtype'):
            intervener_urn = cp.array(intervener_urn, dtype=self.graph.node_urns.dtype)

        if intervener_urn.shape[0] != self.graph.num_colors:
            raise ValueError(f"Urn size mismatch. Expected {self.graph.num_colors}, got {intervener_urn.shape[0]}")

        #add intervener node contents to num_injections next highest degree nodes
        dst = self.degree_ordered_nodes[self.current_num_interventions:self.current_num_interventions + num_injections]
        self.graph.node_urns[dst] += intervener_urn
        # add num injections to num interventions so that we inject on subsequent highest nodes on future injections
        if (self.once_per_node_iv):
            self.current_num_interventions += num_injections
        #print(f" injected intervention urn: {intervener_urn} into nodes: {dst}")
        

