from ..cupy_fallback import cp
from .graph_x import Graph

class Switch_Network_Growth:
    """
    Growth of a graph according to threshold / switched network behavior.
    """
    def __init__(self, graph: Graph, connection_prob=0.01):
        self.graph = graph
        self.connection_prob = connection_prob
        self.current_node = 0  # next node id to add


    def grow_batch(self, batch_size=1000):
        """
        Add a batch of new nodes to the graph.
        Each new node that passes its indep bernoulli trial connects to
        all nodes earlier in the same batch, and all previous nodes in the graph.
        """
        new_nodes = cp.arange(self.current_node, self.current_node + batch_size)
        prev_nodes = cp.arange(self.current_node)
        prev_n = int(prev_nodes.size)

        # Bernoulli trial per new node
        connect_flags = cp.random.rand(batch_size) < self.connection_prob
        if not cp.any(connect_flags):
            self.current_node += batch_size
            return

        # Indices of nodes that passed
        success_idx = cp.nonzero(connect_flags)[0]
        success_nodes = new_nodes[success_idx]
        M = int(success_nodes.size)

        # Connect to all prev nodes
        if prev_n > 0:
            new_ids_prev = cp.repeat(success_nodes, prev_n)
            prev_ids_prev = cp.tile(prev_nodes, M)
            self.graph.add_edges(new_ids_prev, prev_ids_prev)

        # Connect to all earlier nodes in our batch
        if M > 1:
            # Create M x M upper-triangular mask (excluding diagonal)
            mask = cp.triu(cp.ones((M, M), dtype=cp.bool_), k=1)
            # Get coordinates of edges: row = src, col = target
            row_idx, col_idx = cp.nonzero(mask)
            new_ids_intra = success_nodes[row_idx]
            prev_ids_intra = success_nodes[col_idx]
            self.graph.add_edges(new_ids_intra, prev_ids_intra)

        # Advance global node counter
        self.current_node += batch_size

    # alternative grow_batch method
    # note that this batch growing isnt a strict switched network, since nodes in the same batch cannot yet connect to eachother
    """
    def grow_batch(self, batch_size=1000):
        
        #Add a batch of new nodes to the graph.
        #Each node connects to all previous nodes with probability `connection_prob`.
        
        # Construct new array of nodes with ids 1 incremented from last highest
        new_nodes = cp.arange(self.current_node, self.current_node + batch_size) # 1 x current_node + batch_size
        prev_nodes = cp.arange(self.current_node) # 1 x current_node

        # Connection decisions for all new nodes (vectorized), but process previous
        # nodes in column-chunks to reduce peak memory usage for very large graphs.
        if prev_nodes.size > 0:
            col_chunk = 2048
            prev_n = int(prev_nodes.size)
            for col_start in range(0, prev_n, col_chunk):
                col_end = min(prev_n, col_start + col_chunk)
                chunk_size = col_end - col_start

                # generate random matrix of batch_size x chunk_size 
                probs = cp.random.rand(batch_size, chunk_size)
                connections = probs < self.connection_prob  # create bool array batch_size x chunk_size

                # Find connection pairs and append in bulk to the graph to avoid Python loops and 
                # get coords of nonzero entries 
                new_node_indices, prev_node_indices = cp.nonzero(connections)
                if new_node_indices.size > 0:
                    new_node_ids = new_nodes[new_node_indices]
                    # map chunk-local prev indices to global prev node ids
                    prev_node_ids = prev_nodes[col_start + prev_node_indices]
                    self.graph.add_edges(new_node_ids, prev_node_ids)

        self.current_node += batch_size
"""
    # once graph is grown completely represent it as a CSR for efficiency
    def finalize_growth(self):
        self.graph.build_csr()
