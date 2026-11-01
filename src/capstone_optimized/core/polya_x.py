from ..cupy_fallback import cp
from .graph_x import Graph
from . import csr_kernels

class Polya_Process:
    """
    Fully vectorized Polya process for all nodes.
    """
    def __init__(self, graph: Graph, delta=None, memory_enabled = False, memory_decay_time=0, mem_decay = None, remove_initial_urns = False):
        self.graph = graph
        self.step_count = 1
        if delta is None:
            self.delta = cp.ones((self.graph.num_nodes, self.graph.num_colors), dtype=cp.int32)
        else:
            self.delta = delta
        self.memory_enabled = memory_enabled
        self._mem_bits = None
        if self.memory_enabled != False:
            self.memory_decay_time = memory_decay_time
            # With two colours on the GPU the last M draws are kept as bits (9 B per node at M = 72 instead of
            # 288 B: 0.6 GB at 71.6M nodes instead of 20.6 GB, which does not fit). memory_arr still reads and
            # assigns as an (M, n) array of colours, so code that extends it keeps working
            if (csr_kernels.urns_supported(self.graph.node_urns) and self.graph.num_colors == 2
                    and self.memory_decay_time > 0):
                self._mem_bits = cp.zeros((self.memory_decay_time, (self.graph.num_nodes + 31) // 32), dtype=cp.uint32)
                self._mem_n = self.graph.num_nodes
            else:
                self.memory_arr = cp.zeros((self.memory_decay_time, self.graph.num_nodes), dtype=cp.int32)
            self.remove_initial_urns = remove_initial_urns
            if self.remove_initial_urns != False:
                self.initial_urns = self.graph.node_urns.copy()
            if mem_decay is None:
                self.mem_decay = cp.ones((self.graph.num_nodes, self.graph.num_colors), dtype=cp.int32)
            else:
                self.mem_decay = mem_decay


    def step(self):
        draws = self._fused_draws()  # GPU fast path, None when it does not apply
        if draws is None:
            draws = self._draws()

        # Vectorized urn update
        if csr_kernels.urns_supported(self.graph.node_urns) and self._delta_supported():
            # node_urns[i, draws[i]] += delta[i, draws[i]] in one kernel, no index arrays
            csr_kernels.apply_draws(draws, self.delta, self.graph.node_urns)
        else:
            rows = cp.arange(self.graph.num_nodes)
            #print(f"BEFORE: {self.graph.node_urns[1]}") #debug
            #print(f"DRAWS: {draws[1]}") #debug
            #print(f"ADDING: {self.delta[1, draws]} to node 1") #debug
            self.graph.node_urns[rows, draws] += self.delta[rows, draws]
            #print(f"AFTER: {self.graph.node_urns[1]}") #debug

        if self.memory_enabled:
            self.memory_step(draws)
        self.step_count += 1

    def _fused_draws(self):
        """
        Two colours on a graph holding an unweighted CSR (Graph.build_csr): gather the mega urns and draw in
        one kernel (csr_kernels.py). Same random numbers and float64 arithmetic as _draws, so same draws.
        """
        g = self.graph
        if (g.num_colors != 2 or getattr(g, 'csr_indices', None) is None
                or type(g).get_mega_urns is not Graph.get_mega_urns or not csr_kernels.urns_supported(g.node_urns)):
            return None
        rand_vals = cp.random.rand(g.num_nodes)
        return csr_kernels.draw_two_colour(g.csr_indptr, g.csr_indices, g.node_urns, rand_vals)

    def _delta_supported(self):
        return isinstance(self.delta, cp.ndarray) and self.delta.shape == self.graph.node_urns.shape

    def _draws(self):
        mega = self.graph.get_mega_urns()  # num_nodes x num_colors)
        # Compute sums of balls in each mega urn
        row_sums = cp.sum(mega, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # if mega urn is zero, avoid div by zero
        # Create probability matrix holding color probabilities for each node's mega urn
        #cp.float64 precision 
        probs = mega / row_sums  #(num_nodes x num_colors)

        # Vectorized sampling
        # Generate random vals for each node
        #cp.float64 precision 
        rand_vals = cp.random.rand(self.graph.num_nodes) # num_nodes x 1
        if self.graph.num_colors == 2:
            draws = (rand_vals > probs[:, 0]).astype(cp.int32)
        else:
            # array of n nodes with probability of first colour (0<= p <= 1) as col 1, 1.0 as col 2
            cumulative = cp.cumsum(probs, axis=1) # num_nodes x num_colors
            # array of drawn colors for each node based on random val and cumulative probs
            # (if sum = 0 choose col 0's colour, if sum = 1 choose col1's colour)
            draws = cp.sum(rand_vals > cumulative, axis=1) # num_nodes x 1
            # Guard against occasional floating-point rounding producing an index 
            # equal to `num_colors` (2 in our case, still OOB). Clip to valid range.
            # ie if both probs hit, just take the last color (1 in 2-colour case)
            draws = cp.minimum(draws, self.graph.num_colors - 1)
        return draws

    @property
    def memory_arr(self):
        if self._mem_bits is not None:
            return csr_kernels.unpack_bit_rows(self._mem_bits, self._mem_n)
        return self._memory_arr

    @memory_arr.setter
    def memory_arr(self, arr):
        if self._mem_bits is not None:
            self._mem_bits = csr_kernels.pack_bit_rows(arr)
            self._mem_n = arr.shape[1]
        else:
            self._memory_arr = arr

    def memory_step(self, draws):
        if self._mem_bits is not None:
            if self._mem_n != self.graph.num_nodes:
                raise ValueError(f"memory holds {self._mem_n} nodes but the graph has {self.graph.num_nodes}")
            if csr_kernels.urns_supported(self.graph.node_urns):
                if (self.step_count == self.memory_decay_time) and self.remove_initial_urns != False:
                    self.graph.node_urns -= self.initial_urns
                # expire the draw from M steps ago and record this step's, in one kernel
                csr_kernels.memory_bits_step(self.graph.node_urns, draws, self.mem_decay,
                                             self._mem_bits[self.step_count % self.memory_decay_time],
                                             self.step_count > self.memory_decay_time)
                return
            # urns no longer in a form the kernel reads: continue with the array version
            self._memory_arr, self._mem_bits = self.memory_arr, None
        if ((self.step_count == self.memory_decay_time) and self.remove_initial_urns != False):
            # when we hit memory decay time, delete all initial urns from memory 
            self.graph.node_urns -= self.initial_urns
        
        rows = cp.arange(self.graph.num_nodes)
        #if we are at the point where we are removing balls due to memory, find the draws of current step - memory, and subtract their deltas from the urns
        if self.step_count > self.memory_decay_time:
            self.graph.node_urns[rows, self.memory_arr[self.step_count % self.memory_decay_time]] -= self.mem_decay[rows, self.memory_arr[self.step_count % self.memory_decay_time]]
        #add current draws to memory
        self.memory_arr[self.step_count % self.memory_decay_time] = draws
        return