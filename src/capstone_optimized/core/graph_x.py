from ..cupy_fallback import cp, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    from cupyx.scipy.sparse import csr_matrix
else:
    from scipy.sparse import csr_matrix

class Graph:
    """
    Extreme high performance upgrade for graph class with GPU compatibility for matmuls
    After graph is grown store as a CSR adjacency matrix
    Graph allows for directed edges, forces all edge weights to 1 to save on mem. (can change in future)
    Edges are stored as a buffer that grows based on mem availability, and number of edges in the graph
    """
    def __init__(self, num_nodes, num_colors=2, initial_edge_capacity=1_000_000, use_gpu=True):
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        if use_gpu and not CUPY_AVAILABLE:
            print('GPU requested but not available, running on CPU with NumPy')
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        """
        Dynamic edge buffer. Start with some initial capacity for the edges stored in cupy arr.
        Allows for nodes to be added in batches, with edges determined probabilistically.
        As number of edges increases the buffer dynamically grows 
        """
        self.edge_capacity = initial_edge_capacity
        self.edges = cp.zeros((self.edge_capacity, 2), dtype=cp.int32)
        self.num_edges = 0
        # vector holding the urn for each node in the graph
        self.node_urns = cp.ones((num_nodes, num_colors), dtype=cp.int32)
        # most efficient storage possible for edges using CSR adjacency matrix, 
        self.adj_matrix = None

    # add a single edge 
    def add_edge(self, i, j):
        # call bulk append for a single elem
        self.add_edges(cp.array([i], dtype=cp.int32), cp.array([j], dtype=cp.int32))

    """
    Bulk append multiple edges. i_arr and j_arr must be 1D arrays or lists (use numpy or cp lists for efficeincy).
    Avoids expensive Python loops and makes use of GPU matmul.
    """
    def add_edges(self, i_arr, j_arr):
    
        # convert to cp arrays if necessary
        # check if array is numpy/cupy array, if not convert it to one
        if not hasattr(i_arr, 'dtype'):
            i_arr = cp.array(i_arr, dtype=cp.int32)
        else:
            # force type to cupy.int32
            i_arr = i_arr.astype(cp.int32)
        # check if array is numpy/cupy array, if not convert it to one
        if not hasattr(j_arr, 'dtype'):
            j_arr = cp.array(j_arr, dtype=cp.int32)
        else:
            # force type to cupy.int32
            j_arr = j_arr.astype(cp.int32)


        k = int(i_arr.size)
        # Resize buffers if needed. Try to grow conservatively and respect
        # available GPU memory. Attempt an allocation, and if OOM back off
        # instead of unbounded doubling which can overshoot device memory.
        needed = self.num_edges + k
        # if we need more mem to store edges, do the allocation process
        if needed > self.edge_capacity:
            # bytes per row estimate edges has 2 ints
            bytes_per_row = int(self.edges.dtype.itemsize) * 2

            # start with a modest increment strategy: add either k or 25% of
            # current capacity, whichever is larger
            increment = max(k, max(1024, int(self.edge_capacity * 0.25)))
            new_capacity = self.edge_capacity

            # determine a target capacity at least large enough for needed mem
            while new_capacity < needed:
                new_capacity = new_capacity + increment
                increment = min(increment * 2, 10_000_000)

            # If cupy is available, calculate available memory, then use it to cap memory that we can request
            # stops us from overallocating more than the 6gb vram limit.
            # however we slow way down if we are at the limit since GPU starts offloading to unified mem over PCIe
            # which is super slow. TODO: Need to find a better way to handle this in future.
            if CUPY_AVAILABLE:
                try:
                    free, total = cp.cuda.runtime.memGetInfo()
                    # keep a safety margin of 1MB
                    max_additional_rows = max(0, int((free - 1_000_000) // bytes_per_row))
                    if max_additional_rows > 0:
                        max_possible = self.edge_capacity + max_additional_rows
                        if max_possible < new_capacity:
                            # cap new_capacity to what's feasible
                            new_capacity = max(max_possible, needed)
                except Exception:
                    # If mem info not available, proceed with computed new_capacity
                    pass

            # Attempt allocation, on OOM back off progressively until we either
            # succeed or fall back to the minimal required capacity (needed).
            while True:
                try:
                    new_edges = cp.empty((new_capacity, 2), dtype=self.edges.dtype)
                    # copy existing content
                    new_edges[: self.edge_capacity] = self.edges
                    # replace buffers
                    self.edges = new_edges
                    self.edge_capacity = new_capacity
                    break
                except Exception as exc:
                    # If this looks like an OOM, try to reduce growth and retry.
                    msg = str(exc).lower()
                    if 'out of memory' in msg or isinstance(exc, MemoryError) or ('cuda' in msg and 'memory' in msg):
                        # If we've already pared down to just enough, re-raise
                        if new_capacity <= needed:
                            raise
                        # reduce by half of the extra portion, but keep at least needed
                        extra = new_capacity - self.edge_capacity
                        reduce_by = max(int(extra // 2), k)
                        new_capacity = max(self.edge_capacity + k, new_capacity - reduce_by)
                        # continue and retry
                        continue
                    else:
                        # unexpected error, re-raise
                        raise

        start = self.num_edges
        end = start + k
        self.edges[start:end, 0] = i_arr
        self.edges[start:end, 1] = j_arr
        self.num_edges = end

    # TODO: make a CPU version for this since we crash on CUDA malloc if out of vram
    def build_csr(self):
        if self.num_edges == 0:
            self.adj_matrix = None
        else:
            rows = self.edges[: self.num_edges, 0]
            cols = self.edges[: self.num_edges, 1]
            nnz = int(self.num_edges)
            self.adj_matrix = csr_matrix(
                    (cp.ones(nnz, dtype=cp.float32), (rows, cols)),
                    shape=(self.num_nodes, self.num_nodes),
                )
    
    # Urns are initialized as 1, 1 by default, can set them to custom values
    def set_node_urns(self, urns):
        # convert to cp array if necessary
        if not hasattr(urns, 'dtype'):
            urns = cp.array(urns, dtype=cp.int32)
        else:
            urns = urns.astype(cp.int32)
        if urns.shape != (self.num_nodes, self.num_colors):
            raise ValueError(f'urns shape must be ({self.num_nodes}, {self.num_colors}), got {urns.shape}')
        self.node_urns = urns

    def get_mega_urns(self, include_self=True):
        if self.adj_matrix is None:
            # If no edges, each mega urn is just the node's own urn
            return self.node_urns.copy()
        mega_urns = self.adj_matrix @ self.node_urns # nxn * nxc = nxc
        if include_self:
            # Add own urn (vectorized, element-wise)
            mega_urns += self.node_urns
        return mega_urns
