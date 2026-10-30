import numpy as np
from ..cupy_fallback import cp, CUPY_AVAILABLE
from . import csr_kernels
if CUPY_AVAILABLE:
    from cupyx.scipy.sparse import csr_matrix as cp_sparse
    from scipy.sparse import csr_matrix as sp_sparse
else:
    from scipy.sparse import csr_matrix as cp_sparse

class Graph:
    """
    TODO: need to refactor this class to ensure storage happens in RAM, virtual MEM
    Extreme high performance upgrade for graph class with GPU compatibility for matmuls
    After graph is grown store as a CSR adjacency matrix
    Graph allows for directed edges, forces all edge weights to 1 to save on mem. (can change in future)
    Edges are stored as a buffer that grows based on mem availability, and number of edges in the graph
    """
    def __init__(self, num_nodes, node_urns = None, num_colors=2, initial_edge_capacity=1_000_000, use_gpu=True):
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if use_gpu and not CUPY_AVAILABLE:
            print('GPU requested but not available, running on CPU with NumPy')
        
        """
        Dynamic edge buffer. Start with some initial capacity for the edges stored in cupy arr.
        Allows for nodes to be added in batches, with edges determined probabilistically.
        As number of edges increases the buffer dynamically grows 
        """
        self.edge_capacity = initial_edge_capacity
        self.edges = cp.zeros((self.edge_capacity, 2), dtype=cp.int32)
        self.num_edges = 0

        # vector holding the urn for each node in the graph
        if node_urns is None: self.node_urns = cp.ones((num_nodes, num_colors), dtype=cp.int32)
        else: self.node_urns = node_urns

        # most efficient storage possible for edges using CSR adjacency matrix,
        # on the GPU an all-ones CSR is kept as just indptr + indices (see build_csr / csr_kernels.py)
        self.csr_indptr = None
        self.csr_indices = None
        # where build_csr puts the CSR indices: 'auto' (VRAM if it fits), 'device', or 'host' (pinned RAM,
        # read by the GPU over PCIe, as fast here because the step is limited by the random urn reads)
        self.csr_placement = 'auto'
        self.adj_matrix = None

    @property
    def adj_matrix(self):
        # Built on first use from indptr/indices for code that wants a sparse matrix object; the Polya
        # step does not need it, and the all-ones values array would cost as much memory as the indices
        if self._adj_matrix is None and self.csr_indices is not None:
            indices = self.csr_indices if isinstance(self.csr_indices, cp.ndarray) else cp.asarray(self.csr_indices)
            ones = cp.ones(indices.size, dtype=cp.float32)
            self._adj_matrix = cp_sparse((ones, indices, self.csr_indptr), shape=(self.num_nodes, self.num_nodes))
        return self._adj_matrix

    @adj_matrix.setter
    def adj_matrix(self, matrix):
        # a matrix assigned directly is used as-is (original SpMM path)
        self._adj_matrix = matrix
        self.csr_indptr = None
        self.csr_indices = None

    def release_edge_buffer(self):
        # The edge buffer is only needed to (re)build the CSR. Once growth is finished it is the largest
        # array on the GPU, and if it stays allocated the Polya step's arrays get pushed into shared memory
        self.edges = None
        self.edge_capacity = 0
        if CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

    # add a single edge 
    def add_edge(self, i, j):
        # call bulk append for a single elem
        self.add_edges(cp.array([i], dtype=cp.int32), cp.array([j], dtype=cp.int32))

    """
    Bulk append multiple edges. i_arr and j_arr must be 1D arrays or lists (use numpy or cp lists for efficeincy).
    Avoids expensive Python loops and makes use of GPU matmul.
    """
    def add_edges(self, i_arr, j_arr):
        if self.edges is None:
            raise RuntimeError('edge buffer was released after the CSR was built (finalize_growth), cannot add edges')

        # convert to cp arrays if necessary
        # check if array is numpy/cupy array, if not convert it to one
        if not hasattr(i_arr, 'dtype'): i_arr = cp.array(i_arr, dtype=cp.int32)
        else: i_arr = i_arr.astype(cp.int32)

        if not hasattr(j_arr, 'dtype'): j_arr = cp.array(j_arr, dtype=cp.int32)
        else: j_arr = j_arr.astype(cp.int32)

        k = int(i_arr.size)

        # Resize buffers if needed. 
        needed = self.num_edges + k
        # if we need more mem to store edges, do the allocation process
        if needed > self.edge_capacity:
            # bytes per row estimate edges has 2 ints
            bytes_per_row = int(self.edges.dtype.itemsize) * 2
            # start with a modest increment strategy: add either # new edges or 25% of
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
        # custom gather kernel: same values as the SpMM below (float64), without its float64 copies of A and U
        if (self.csr_indices is not None and self.num_colors <= csr_kernels.MAX_COLORS
                and csr_kernels.urns_supported(self.node_urns)):
            return csr_kernels.mega_urns(self.csr_indptr, self.csr_indices, self.node_urns, include_self)

        # If no edges, each mega urn is just the node's own urn
        if self.adj_matrix is None:
            return self.node_urns.copy()

        mega_urns = self.adj_matrix @ self.node_urns # nxn * nxc = nxc
        if include_self:
            # Add own urn (vectorized, element-wise)
            mega_urns += self.node_urns
        return mega_urns
    
    def build_csr(self, release_edges=False):
        # release_edges: free the edge buffer once it has been read (end of growth, see finalize_growth)
        if self.num_edges == 0:
            self.adj_matrix = None
        else:
            # if gpu available and we have 10M + edges build csr on cpu instead, much faster than on gpu because of memory availability.
            if self.num_edges > 10**6 and self.use_gpu:
                print(f"Building CSR on CPU then moving to GPU")
                rows = cp.asnumpy(self.edges[: self.num_edges, 0])
                cols = cp.asnumpy(self.edges[: self.num_edges, 1])
                if release_edges:
                    self.release_edge_buffer()
                nnz = np.ones(int(self.num_edges), dtype=np.float32)
                cpu_csr = sp_sparse((nnz, (rows, cols)), shape=(self.num_nodes, self.num_nodes),)
                del rows, cols, nnz

                # then move back to vram (or pinned RAM if the indices do not fit, see _store_csr)
                if not self._store_csr(cpu_csr.indptr, cpu_csr.indices, cpu_csr.data):
                    gpu_data = cp.array(cpu_csr.data)
                    gpu_indices = cp.array(cpu_csr.indices)
                    gpu_indptr = cp.array(cpu_csr.indptr)
                    self.adj_matrix = cp_sparse((gpu_data, gpu_indices, gpu_indptr), shape=cpu_csr.shape)

            else: #standard build procedure, if cp not available uses np
                print(f"Building CSR on {'GPU' if self.use_gpu else 'CPU'}")
                rows = self.edges[: self.num_edges, 0]
                cols = self.edges[: self.num_edges, 1]
                nnz = cp.ones(int(self.num_edges), dtype=cp.float32)
                adj = cp_sparse((nnz, (rows, cols)), shape=(self.num_nodes, self.num_nodes),)
                del rows, cols, nnz
                if release_edges:
                    self.release_edge_buffer()
                if not self._store_csr(adj.indptr, adj.indices, adj.data):
                    self.adj_matrix = adj

    def _store_csr(self, indptr, indices, data):
        """
        Keep only indptr/indices of an all-ones CSR for the custom step kernels (csr_kernels.py).
        Returns False, storing nothing, if the matrix has other weights (duplicate edges summed by the
        CSR build) or needs 64-bit indices; the original sparse matrix path is used then.
        """
        if not self.use_gpu or indptr.dtype != np.int32 or indices.dtype != np.int32:
            return False
        if data.size and not (data.min() == 1 and data.max() == 1):
            return False
        self.adj_matrix = None
        self.csr_indptr = cp.asarray(indptr)
        if self._csr_indices_on_host(indices.size):
            print(f"CSR indices do not fit in VRAM, keeping them in pinned host memory")
            pinned = cp.cuda.alloc_pinned_memory(max(indices.nbytes, 1))
            host = np.frombuffer(pinned, dtype=np.int32, count=indices.size)
            host[:] = indices if isinstance(indices, np.ndarray) else cp.asnumpy(indices)
            self.csr_indices = host
        else:
            self.csr_indices = cp.asarray(indices)
        return True

    def _csr_indices_on_host(self, nnz):
        if self.csr_placement != 'auto':
            return self.csr_placement == 'host'
        # VRAM the Polya step needs on top of what is allocated now: the indices, plus per node the
        # float64 random values, int32 draws and an int32 delta row
        need = 4 * nnz + self.num_nodes * (8 + 4 + 4 * self.num_colors)
        cp.get_default_memory_pool().free_all_blocks()
        free, _ = cp.cuda.runtime.memGetInfo()
        return need > 0.9 * free - 256 * 2**20
