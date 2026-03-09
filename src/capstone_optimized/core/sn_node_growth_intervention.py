from ..cupy_fallback import cp
from .graph_x import Graph

class Switch_Network_Node_Growth_Intervention:
    """
    Add new red biased nodes to the graph that will connect to the highest degree nodes.
    This is "Method 1"
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        self.current_num_interventions = 0
        # information about the edge buffer and degrees of each node in the graph
        self.active_buffer = self.graph.edges[:self.graph.num_edges].flatten()
        self.degrees = cp.bincount(self.active_buffer, minlength=self.graph.num_nodes)
        self.degree_ordered_nodes = cp.flip(cp.argsort(self.degrees))
        #print(f"degree_ordered_nodes: {self.degree_ordered_nodes}")

    def degree_centrality_optimized_intervention_step(self, num_connections=1, intervention_deltas=1, intervener_urn=None):
        """
        At each intervention step, we create a new node on the graph with num_connections # edges
        connecting to existing nodes. Each connected node's mega urn will thus include intervener_urn, and
        the new intervener node will have a node_urn defined by intervener_urn.

        This function will select targets for the intervener node based on degree centrality, connecting to
        the #num_connections highest degree nodes on the graph in descending order
        """
        # cast intervener_urn if necessary
        if intervener_urn is None:
            intervener_urn = cp.ones((1, self.graph.num_colors), dtype=cp.int32)
        if not hasattr(intervener_urn, 'dtype'): intervener_urn = cp.array(intervener_urn, dtype=self.graph.node_urns.dtype)
        new_node_id = self.graph.num_nodes
        self.graph.num_nodes += 1

        if intervener_urn.shape[0] != self.graph.num_colors:
            raise ValueError(f"Urn size mismatch. Expected {self.graph.num_colors}, got {intervener_urn.shape[0]}")

        #add intervener node to existing list
        self.graph.node_urns = cp.vstack([self.graph.node_urns, intervener_urn])

        # connect new node to best available choices
        src = cp.full(num_connections, new_node_id, dtype=cp.int32)
        dst = self.degree_ordered_nodes[self.current_num_interventions:self.current_num_interventions + num_connections]

        #increment counter so we know not to reconnect to the same nodes
        self.current_num_interventions += num_connections

        # add edges
        self.graph.add_edges(src, dst)
        self.graph.add_edges(dst, src)

        #print(f" injected intervention node {new_node_id} connected to nodes {dst} ")
        #print(f" Target Hub IDs: {top_hubs}")
            
    #djeffrey TODO
    #def manual_intervention_step(self, intervention_nodes):
        

