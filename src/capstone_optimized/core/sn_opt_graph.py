from ..cupy_fallback import cp, CUPY_AVAILABLE
from .graph_x import Graph

class Switched_Network_Optimized_Graph (Graph):
    """
    Maximally optimized graph class exclusively for switched network style graphs.
    
    NOTE: This will NOT WORK for any general growth process, it relies on the fact 
    that in a switched network growth model newly generated nodes connect either to 
    all previous nodes or none at all, this makes the adjacency matrix a set of symmetric L's 
    of 1's at rows that hit the switch condition (plus the identity matrix).

    [1, 0, 1, 0, 0, 0, 1, 0;    # generated as 0
     0, 1, 1, 0, 0, 0, 1, 0;    # generated as 0
     1, 1, 1, 0, 0, 0, 1, 0;    # connects to all prev
     0, 0, 0, 1, 0, 0, 1, 0;    # generated as 0
     0, 0, 0, 0, 1, 0, 1, 0;    # generated as 0
     0, 0, 0, 0, 0, 1, 1, 0;    # generated as 0
     1, 1, 1, 1, 1, 1, 1, 0;    # connects to all prev
     0, 0, 0, 0, 0, 0, 0, 1;]   # generated as 0

    because the matrix is structured this way we can instead just define it as a 1xN vector
    of success flags indicating which nodes connected to all previous nodes, since this vector
    encodes all the information of the adjacency matrix.
    """
    def __init__(self, num_nodes, num_colors=2, use_gpu=True):
        super().__init__(num_nodes, num_colors, initial_edge_capacity=0, use_gpu=use_gpu)
        # success_flags[i] = 1 if node i connects to all previous nodes
        self.success_flags = cp.zeros(num_nodes, dtype=cp.bool_)
        self.edges = None  # disable edge buffer
        self.edge_capacity = cp.uint64(0)

    # Overrided add_edges, only need 1xN vector of new nodes indices that passed switch condition
    # Also no longer need the OOM checks since we arent storing any edges
    def add_edges(self, i_arr, j_arr=None):
        if not hasattr(i_arr, 'dtype'):
            i_arr = cp.array(i_arr, dtype=cp.uint64)
        else:
            i_arr = i_arr.astype(cp.uint64)
        self.success_flags[i_arr] = True
        # Track number of edges for logging, just have to add up the indices of new nodes that succeeded
        # since if node n succeeds it connects to 0, 1, ..., n-1 = n edges
        self.num_edges += cp.sum(i_arr) 

    # Override build_csr to do nothing since we dont need csr representation
    def build_csr(self):
        self.adj_matrix = None

    def get_mega_urns(self, include_self=True):
        # element-wise mult 
        success_urns = self.node_urns * self.success_flags[:, None] # nx2 * 1xn = nx2 where only rows that succeeded are nonzero
        failure_urns = self.node_urns * (~self.success_flags[:, None]) # nx2 * 1xn = nx2 where only rows that failed are nonzero
        mega_urns = cp.flip(cp.cumsum(cp.flip(success_urns), axis=0), axis=0) + failure_urns # nx2 cumulative sum of reversed contrib along cols
        return mega_urns

