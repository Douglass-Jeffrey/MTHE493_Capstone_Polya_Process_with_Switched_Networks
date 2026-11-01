from ..cupy_fallback import cp, CUPY_AVAILABLE
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
        
        # The stub list is not stored. Each batch appends [src, dst] to the stubs and the rows (src, dst) then
        # (dst, src) to the graph's edge buffer, so after the initial part stub r is edges[r - shift, 0]. Only
        # the initial part (clique or pre-existing edges, laid out differently) is kept as an array, _stubs_head.
        # This saves a buffer as large as the edge list's first column (1.7 GB at 71.6M nodes, plus slack)
        self.stubs = None
        self.num_stubs = 0
        self._stubs_head = None
        self._stub_shift = 0
        # free CuPy's cache of outgrown buffers when VRAM runs low (see grow_batch); False keeps the old
        # behaviour, where the cache grows until the driver moves live arrays into system RAM
        self.free_cache = True

        if self.graph.num_edges > 0:
            # Unlikely i will want to do BA growth on a pre-existing graph, but just in case
            active_edges = self.graph.edges[:self.graph.num_edges]
            self._stubs_head = active_edges.flatten().astype(cp.int32)
            self.num_stubs = self._stubs_head.size
            self._stub_shift = self.num_stubs - self.graph.num_edges

    def reserve(self, final_num_nodes):
        """
        Size the graph's edge buffer for a network grown to final_num_nodes, so it is never resized. A resize
        holds the old and the new buffer at once, at 71.6M nodes 3.4 GB and 4.3 GB, more than the GPU has.
        Only done if the buffer and the urns fit in free VRAM: a buffer the driver has to keep in system RAM
        from the start makes every batch read and write it over PCIe. Returns whether it reserved.
        """
        init = max(self.m, self.initial_nodes) if self.graph.num_edges == 0 else 0
        rows = self.graph.num_edges + init * (init - 1) + 2 * self.m * max(0, final_num_nodes - max(init, self.graph.num_nodes))
        if rows <= self.graph.edge_capacity:
            return True
        if CUPY_AVAILABLE:
            # the urns are replaced (copied) every batch by the callbacks' vstack, so count them twice
            need = (rows - self.graph.edge_capacity) * 8 + 2 * final_num_nodes * self.graph.num_colors * 4
            cp.get_default_memory_pool().free_all_blocks()
            if need > cp.cuda.runtime.memGetInfo()[0] - 256 * 2**20:
                return False
        new_edges = cp.empty((rows, 2), dtype=self.graph.edges.dtype)
        new_edges[:self.graph.num_edges] = self.graph.edges[:self.graph.num_edges]
        self.graph.edges = new_edges
        self.graph.edge_capacity = rows
        return True

    def _stub_values(self, idx):
        """stubs[idx] for int64 stub indices, read from the head array or the edge buffer."""
        head = self._stubs_head
        tail = self.graph.edges[cp.maximum(idx - self._stub_shift, 0), 0]
        if head is None:
            return tail
        return cp.where(idx < head.size, head[cp.minimum(idx, head.size - 1)], tail)

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

        # stubs: src then dst (each clique edge counted from both ends, as in the stored stub list before)
        self._stubs_head = cp.concatenate([src, dst])
        self.num_stubs = self._stubs_head.size
        self._stub_shift = self.num_stubs - self.graph.num_edges
        
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
        targets = self._stub_values(rand_indices)

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
            targets[bad_indices] = self._stub_values(new_rands)
            
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
        
        # the new stubs are the rows just appended to the edge buffer (see __init__)
        self.num_stubs += src.size + dst.size

        # CuPy caches every buffer this growth has outgrown (the urn arrays replaced by the callback's vstack,
        # old edge buffers), and none of them fits a later, larger request. Left alone the cache
        # reaches ~14 GB at 71.6M nodes and the driver moves live arrays out to system RAM to make room
        # Freeing is only done once VRAM runs low: each cudaFree takes ~20 ms under the Windows display driver
        if CUPY_AVAILABLE and self.free_cache:
            pool = cp.get_default_memory_pool()
            cached = pool.total_bytes() - pool.used_bytes()
            if cached > 256 * 2**20 and cp.cuda.runtime.memGetInfo()[0] < cached + 512 * 2**20:
                pool.free_all_blocks()
        #print ("stubs after growth:\n", self.stubs[:self.num_stubs])

    def finalize_growth(self, release_edges=True):
        # Growth is over: the stub list and the graph's edge buffer are not needed after the CSR is built.
        # Freed here so the Polya step's arrays fit in VRAM (at 71.6M nodes they are 1.6 GB and 3.9 GB).
        # Pass release_edges=False to keep the edge buffer when interventions will add nodes mid-run
        # (Switch_Network_Node_Growth_Intervention with mid_polya_intervention=True appends edges and rebuilds)
        self._stubs_head = None
        self.num_stubs = 0
        self.graph.build_csr(release_edges=release_edges)
        #print("stubs final size:", self.num_stubs)