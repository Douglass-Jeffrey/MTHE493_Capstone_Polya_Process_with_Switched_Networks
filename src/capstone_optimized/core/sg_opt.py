from ..cupy_fallback import cp, CUPY_AVAILABLE
from .switched_growth import Switch_Network_Growth
from .sn_opt_graph import Switched_Network_Optimized_Graph

class Dim_Optimized_Switch_Network_Growth(Switch_Network_Growth):
    """
    Growth of a switched network graph, optimized to use Switched_Network_Optimized_Graph.
    """
    def __init__(self, graph: Switched_Network_Optimized_Graph, connection_prob=0.01):
        super().__init__(graph, connection_prob)

    # Override grow_batch for optimized graph, we just need to send indexes
    # of new nodes that hit switched network condition
    def grow_batch(self, batch_size):

        """
        Add a batch of new nodes to the graph.
        Each new node that passes its indep bernoulli trial connects to
        all nodes earlier in the same batch, and all previous nodes in the graph.
        """
        # Create new node ids
        new_nodes = cp.arange(self.current_node, self.current_node + batch_size)
        # Bernoulli trial per new node
        connect_flags = cp.random.rand(batch_size) < self.connection_prob
        if not cp.any(connect_flags):
            self.current_node += batch_size
            return

        # Indices of nodes that passed
        success_idx = new_nodes[connect_flags]
        self.graph.add_edges(success_idx)

        return connect_flags