from ..cupy_fallback import cp
from .graph_x import Graph

class Polya_Process:
    """
    Fully vectorized Polya process for all nodes.
    """
    def __init__(self, graph: Graph, delta=None, memory_enabled = False, memory_decay_time=0):
        self.graph = graph
        self.step_count = 1
        if delta == None:
            self.delta = cp.ones((self.graph.num_nodes, self.graph.num_colors), dtype=cp.int32)
        else:
            self.delta = delta
        self.memory_enabled = memory_enabled
        if self.memory_enabled != False:
            self.memory_decay_time = memory_decay_time
            self.memory_arr = cp.zeros((self.memory_decay_time, self.graph.num_nodes), dtype=cp.int32)
            self.initial_urns = self.graph.node_urns.copy()


    def step(self):
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

        # Vectorized urn update 
        rows = cp.arange(self.graph.num_nodes)
        self.graph.node_urns[rows, draws] += self.delta[rows, draws]

        if self.memory_enabled:
            self.memory_step(draws)
        self.step_count += 1
        
    def memory_step(self, draws):
        if self.step_count < self.memory_decay_time:
            self.memory_arr[self.step_count] = draws
        else:
            if self.step_count == self.memory_decay_time:
                # when we hit memory decay time, delete all initial urns from memory 
                self.graph.node_urns -= self.initial_urns
            else: 
                rows = cp.arange(self.graph.num_nodes)
                #if we are at the point where we are removing balls due to memory, find the draws of current step - memory, and subtract their deltas from the urns 
                self.graph.node_urns[rows, self.memory_arr[self.step_count % self.memory_decay_time]] -= self.delta[rows, self.memory_arr[self.step_count % self.memory_decay_time]]
                #add current draws to memory
                self.memory_arr[self.step_count % self.memory_decay_time] = draws
        return