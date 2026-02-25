from ..cupy_fallback import cp
from .graph_x import Graph

class Barabasi_Albert_Growth:
    """
    Optimized Vectorized Barabási–Albert growth using Cupy.
    
    Uses the 'Stub List' method for O(1) preferential attachment sampling.
    Instead of calculating probabilities for every node (which is O(N)), 
    we maintain a list of 'stubs' (existing edge endpoints). Sampling uniformly 
    from this list is mathematically equivalent to sampling by degree.
    However we do not have intra batch connections, so not a perfect BA model.
    """

    def __init__(self, graph: Graph, m: int = 2, initial_nodes: int = 5):
        self.graph = graph
        self.m = m
        self.current_nodes = self.graph.num_nodes
        self.initial_nodes = initial_nodes
        # Create stub list for preferential attachment
        
        # Take existing list of edges, flatten into 1D array of stubs,
        # because edges stored pairs of (src, dst), by flattening we get
        # a list where each node appears as many times as its degree.
        # we can then sample uniformly from this list to make new
        # connections according to preferential attachment. (Molloy-Reed Trick)
        
        # we are also implementing auto growing of the stubs array as needed
        # given that we are trying to represent 10**8 or more nodes
        initial_capacity = max(1_000_000, self.graph.num_nodes)
        self.stubs_capacity = initial_capacity
        self.stubs = cp.zeros(self.stubs_capacity, dtype=cp.int32)
        self.num_stubs = 0

        if self.graph.num_edges > 0:
            active_edges = self.graph.edges[:self.graph.num_edges]
            flat_edges = active_edges.flatten().astype(cp.int32)
            count = flat_edges.size

            # Unlikely i will want to do BA growth on a pre-existing graph, but just in case
            # if number of edges flattened exceeds capacity, grow stubs array accordingly
            if count > self.stubs_capacity:
                self.stubs_capacity = count
                self.stubs = cp.zeros(self.stubs_capacity, dtype=cp.int32)

            # then update stubs accordingly
            self.stubs[:count] = flat_edges
            self.num_stubs = count

    def _ensure_stubs_capacity(self, needed_extra):
        """Checks if stubs array needs resizing and handles it efficiently."""
        needed_total = self.num_stubs + needed_extra
        if needed_total > self.stubs_capacity:
            new_cap = max(needed_total, int(self.stubs_capacity * 1.5))
            print(f"Resizing Stubs: {self.stubs_capacity} -> {new_cap}")
            
            new_stubs = cp.empty(new_cap, dtype=self.stubs.dtype)
            new_stubs[:self.num_stubs] = self.stubs[:self.num_stubs]
            self.stubs = new_stubs
            self.stubs_capacity = new_cap

    def _initialize_clique(self, m_init):
        # generate some initial clique of nodes at beginning of growth
        nodes = cp.arange(m_init, dtype=cp.int32)
        row_ids = cp.repeat(nodes, m_init)
        col_ids = cp.tile(nodes, m_init)        
        # remove self-loops
        mask = row_ids != col_ids
        src = row_ids[mask]
        dst = col_ids[mask]
        #print (f"i_src: {src}\n")
        #print (f"i_dst: {dst}\n")
        # add to graph (dont need to add dst->src since those are included in src->dst
        # as implemented above)
        self.graph.add_edges(src, dst)

        # Add to local stubs buffer, check if we need to resize (wont unless our initial clique is 1M+)
        new_stubs_count = src.size + dst.size
        self._ensure_stubs_capacity(new_stubs_count)
        
        end_idx = self.num_stubs + new_stubs_count
        # Fill stubs with new connections(append src and dst)
        self.stubs[self.num_stubs : self.num_stubs + src.size] = src
        self.stubs[self.num_stubs + src.size : end_idx] = dst
        self.num_stubs = end_idx
        
        # update node count if needed
        if self.graph.num_nodes < m_init:
            self.graph.num_nodes = m_init
        #print("stubs after init:", self.stubs[:self.num_stubs])

    """
    Note on Parallelism:
    In standard sequential BA, node t connects to t-1. In batched BA,
    all nodes in the batch connect to the existing pool at the START
    of the batch. This is a standard approximation for GPU performance.
    """
    def grow_batch(self, batch_size: int = 1024, urn_addition_callback = None):
        if batch_size < 1: return

        # Initialization check
        if self.num_stubs == 0 or self.graph.num_nodes < self.m:
            init_size = max(self.m, self.initial_nodes)
            if self.graph.num_edges == 0:
                self._initialize_clique(init_size)
                remaining = (self.graph.num_nodes + batch_size) - init_size
                if remaining <= 0: return
                batch_size = remaining
            
        start_node = self.graph.num_nodes
        end_node = start_node + batch_size

        # Add urns for the batch of new nodes
        if end_node > self.graph.num_nodes:
            # if we are adding some default nodes, just use ones
            if urn_addition_callback is None:
                extra = end_node - self.graph.num_nodes
                extra_urns = cp.ones((extra, self.graph.num_colors), dtype=self.graph.node_urns.dtype)
                self.graph.node_urns = cp.vstack([self.graph.node_urns, extra_urns])
                self.graph.num_nodes = end_node
            #callback should COMPLETELY handle adding the new node urns
            else: urn_addition_callback(self, batch_size)

        # Sampling using stubs for preferential attachment
        # Important: We sample from stubs[:self.num_stubs], effectively ignoring the empty buffer space
        rand_indices = cp.random.randint(0, self.num_stubs, size=(batch_size, self.m), dtype=cp.int64)
        targets = self.stubs[rand_indices]

        # Check if a new node has selected the same target more than once
        sorted_targets = cp.sort(targets, axis=1)
        duplicates = (cp.diff(sorted_targets, axis=1) == 0)
        has_dupes = cp.any(duplicates, axis=1)
        
        # if we have duplicates, resample those entries
        loop_limit = 0
        while cp.any(has_dupes) and loop_limit < 10:
            bad_indices = cp.where(has_dupes)[0]
            num_bad = bad_indices.size
            new_rands = cp.random.randint(0, self.num_stubs, size=(num_bad, self.m), dtype=cp.int64)
            targets[bad_indices] = self.stubs[new_rands]
            
            sorted_targets = cp.sort(targets, axis=1)
            duplicates = (cp.diff(sorted_targets, axis=1) == 0)
            has_dupes = cp.any(duplicates, axis=1)
            loop_limit += 1

        # create edge arrays for new connections
        new_node_ids = cp.arange(start_node, end_node, dtype=cp.int32)
        src = cp.repeat(new_node_ids, self.m)
        dst = targets.flatten()

        #print (f"src: {src}\n")
        #print (f"dst: {dst}\n")

        # Add to Graph
        self.graph.add_edges(src, dst)
        self.graph.add_edges(dst, src)

        # --- Update Stubs (Append new edges to buffer) ---
        # We added (batch * m) edges. Each edge has 2 endpoints (src, dst).
        # Total new entries in stubs = 2 * batch * m * 2 (undirected x2 symmetry)
        # Actually: add_edges is called twice. 
        #   First call: src->dst. Stubs adds src, dst.
        #   Second call: dst->src. Stubs adds dst, src.
        # This results in QUADRUPLE representation which is valid for undirected preferential 
        # attachment math, OR we can just add the logical edges once. 
        # Standard BA: The probability is proportional to degree.
        # If we add (u,v) and (v,u) to the graph, u's degree increases by 1.
        # So u should appear 1 more time in the stubs list.
        #
        # Optimization: We only need to add `src` and `dst` arrays to stubs ONCE 
        # to represent the 1 new degree for each node.
        
        needed_space = src.size + dst.size
        self._ensure_stubs_capacity(needed_space)
        
        current_ptr = self.num_stubs
        self.stubs[current_ptr : current_ptr + src.size] = src
        self.stubs[current_ptr + src.size : current_ptr + src.size + dst.size] = dst
        self.num_stubs += needed_space
        #print ("stubs after growth:\n", self.stubs[:self.num_stubs])

    def finalize_growth(self):
        self.graph.build_csr()
        #print("stubs final size:", self.num_stubs)