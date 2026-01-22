from ..cupy_fallback import cp
from .graph_x import Graph

class Barabasi_Albert_Growth:
    """
    Sequential Barabási–Albert growth integrated with the Graph class.
    Adds one node at a time, each connecting to m existing nodes preferentially.
    Uses Graph.add_edges to handle GPU/CPU dynamic edge buffers efficiently.
    """

    def __init__(self, graph: Graph, m=2, initial_nodes=None):
        self.graph = graph
        self.m = m

        # Track current number of added nodes
        if initial_nodes is None:
            self.current_nodes = 0
        else:
            self.current_nodes = initial_nodes
            if self.current_nodes > self.graph.num_nodes:
                raise ValueError("Initial nodes cannot exceed graph.num_nodes")

        # Initialize degrees array for preferential attachment
        if self.current_nodes > 0:
            self.degrees = cp.zeros(self.current_nodes, dtype=cp.int32)
        else:
            self.degrees = cp.zeros(0, dtype=cp.int32)

    def add_node(self):
        """
        Add a single new node using preferential attachment
        """
        new_node = self.current_nodes
        N = int(self.degrees.size)

        if N == 0:
            # first node, just initialize degree
            self.degrees = cp.array([0], dtype=cp.int32)
            self.current_nodes += 1
            return

        # preferential attachment probabilities
        deg_sum = self.degrees.sum()
        if deg_sum == 0:
            probs = cp.ones(N) / N
        else:
            probs = self.degrees / deg_sum

        # choose m distinct existing nodes
        m_actual = min(self.m, N)
        
        targets = cp.random.choice(N, size=m_actual, replace=False, p=probs)

        # add undirected edges
        new_nodes_arr = cp.full(targets.shape, new_node, dtype=cp.int32)
        self.graph.add_edges(new_nodes_arr, targets)
        self.graph.add_edges(targets, new_nodes_arr)

        # update degrees
        self.degrees = cp.append(self.degrees, m_actual)
        self.degrees[targets] += 1

        # increment current node counter
        self.current_nodes += 1

    def grow(self):
        """
        Grow the graph sequentially until graph.num_nodes is reached
        """
        while self.current_nodes < self.graph.num_nodes:
            self.add_node()

    def finalize_growth(self):
        """
        Build CSR adjacency matrix once after growth is complete
        """
        self.graph.build_csr()

    """
    #Vectorized Barabasi-Albert (BA) growth class for GPU/CPU.
    #Fully vectorized batch growth with preferential attachment.
    
    #Attributes
    #----------
    #graph : Graph
    #    Graph object to grow
    #m : int
    #    Number of edges to attach from each new node to existing nodes
    #current_node : int
    #    ID of the next node to add
    #degrees : cp.ndarray
    #    Degree of each node, used for preferential attachment
    def __init__(self, graph: Graph, m: int = 1):
        self.graph = graph
        self.m = m
        self.current_node = 0
        # initialize degrees
        if graph.num_edges > 0:
            # compute degrees from existing edges
            rows = graph.edges[: graph.num_edges, 0].astype(cp.int32)
            cols = graph.edges[: graph.num_edges, 1].astype(cp.int32)
            deg = cp.bincount(cp.concatenate([rows, cols]), minlength=graph.num_nodes)
            self.degrees = deg.astype(cp.int32)
        else:
            self.degrees = cp.zeros(graph.num_nodes, dtype=cp.int32)

    def grow_batch(self, batch_size: int = 1024):
        #Fully vectorized batch growth for Barabasi-Albert network.
        #- Uses Gumbel-Max trick for weighted sampling without replacement
        #- Fully GPU vectorized: no Python loops
        #- Dynamically grows Graph.num_nodes, node_urns, and degrees
        if batch_size < 1:
            return

        start = self.current_node
        end = start + batch_size

        # --- Expand Graph arrays if needed ---
        old_n = self.graph.num_nodes
        if end > old_n:
            extra = end - old_n
            extra_urns = cp.ones((extra, self.graph.num_colors), dtype=self.graph.node_urns.dtype)
            self.graph.node_urns = cp.vstack([self.graph.node_urns, extra_urns])
            self.graph.num_nodes = end

        # Expand degrees array if needed
        if end > self.degrees.size:
            grow = end - self.degrees.size
            self.degrees = cp.concatenate([self.degrees, cp.zeros(grow, dtype=self.degrees.dtype)])

        current_N = start
        new_nodes = cp.arange(start, end, dtype=cp.int32)

        # If there are existing nodes but no edges yet, create a small
        # initial clique among the first m (or fewer) nodes so that
        # preferential attachment has meaningful targets. This fixes the
        # case where a user initializes the Graph with N nodes but no
        # edges and then grows — otherwise those initial nodes remain
        # isolated and new nodes attach only among themselves.
        if current_N > 0 and getattr(self.graph, 'num_edges', 0) == 0:
            print("hit this")
            # If the user pre-created existing nodes but no edges, fully
            # connect all existing nodes so that preferential attachment
            # uses those nodes as legitimate targets. Use a vectorized
            # construction of the full clique among current_N nodes.
            m_init = int(current_N)
            if m_init > 0:
                idx = cp.arange(m_init, dtype=cp.int32)
                rows = cp.repeat(idx, m_init)
                cols = cp.tile(idx, m_init)
                mask = rows != cols
                rows = rows[mask]
                cols = cols[mask]
                self.graph.add_edges(rows, cols)
                # set degrees for initial nodes
                self.degrees[:m_init] = m_init - 1

        if current_N == 0:
            # Initial batch: connect first m nodes into a clique
            m_init = max(self.m, 1)
            # Vectorized clique: full grid of pairs then remove self-loops
            idx = cp.arange(m_init, dtype=cp.int32)
            rows = cp.repeat(idx, m_init)
            cols = cp.tile(idx, m_init)
            mask = rows != cols
            rows = rows[mask]
            cols = cols[mask]
            self.graph.add_edges(rows, cols)
            self.degrees[:m_init] = m_init - 1
            self.current_node = m_init
            return

        # --- Gumbel-Max sampling ---
        deg_slice = self.degrees[:current_N].astype(cp.float64)
        deg_sum = float(deg_slice.sum())

        # limit m to available existing nodes to avoid empty selections
        m_eff = int(min(self.m, max(0, current_N)))
        if m_eff == 0:
            # nothing to attach to (should only happen if current_N==0 which is handled above)
            chosen = cp.empty((batch_size, 0), dtype=cp.int32)
        elif deg_sum == 0.0:
            # uniform sampling without replacement (vectorized)
            perm = cp.random.permutation(current_N)
            # take first m_eff entries and tile for each new node
            chosen = cp.tile(perm[:m_eff], (batch_size, 1)).astype(cp.int32)
        else:
            P = deg_slice / deg_sum
            logp = cp.log(P + 1e-12)
            gumbel = -cp.log(-cp.log(cp.random.rand(batch_size, current_N) + 1e-12) + 1e-12)
            scores = logp[cp.newaxis, :] + gumbel
            kth = -m_eff
            part = cp.argpartition(scores, kth, axis=1)
            chosen = part[:, kth:].astype(cp.int32)

        # --- Bulk undirected edges ---
        src = cp.repeat(new_nodes, m_eff)
        dst = chosen.reshape(-1)
        self.graph.add_edges(src, dst)
        self.graph.add_edges(dst, src)

        # --- Update degrees ---
        self.degrees[start:end] += m_eff
        if dst.size > 0:
            counts = cp.bincount(dst, minlength=current_N).astype(self.degrees.dtype)
            self.degrees[:current_N] += counts[:current_N]

        # Advance node pointer
        self.current_node = end

    def finalize_growth(self):
        self.graph.build_csr()
    """
