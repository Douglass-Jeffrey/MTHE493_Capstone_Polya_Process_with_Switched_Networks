from ..cupy_fallback import cp
from .graph_x import Graph

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
        if self.memory_enabled != False:
            self.memory_decay_time = memory_decay_time
            self.memory_arr = cp.zeros((self.memory_decay_time, self.graph.num_nodes), dtype=cp.int32)
            self.remove_initial_urns = remove_initial_urns
            if self.remove_initial_urns != False:
                self.initial_urns = self.graph.node_urns.copy()
            if mem_decay is None:
                self.mem_decay = cp.ones((self.graph.num_nodes, self.graph.num_colors), dtype=cp.int32)
            else:
                self.mem_decay = mem_decay


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
        #print(f"BEFORE: {self.graph.node_urns[1]}") #debug
        #print(f"DRAWS: {draws[1]}") #debug
        #print(f"ADDING: {self.delta[1, draws]} to node 1") #debug
        self.graph.node_urns[rows, draws] += self.delta[rows, draws]
        #print(f"AFTER: {self.graph.node_urns[1]}") #debug

        if self.memory_enabled:
            self.memory_step(draws)
        self.step_count += 1
        
    def memory_step(self, draws):
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