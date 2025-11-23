from ..cupy_fallback import cp
from .graph_x import Graph

class Polya_Process:
    """
    Fully vectorized Polya process for all nodes.
    """
    def __init__(self, graph: Graph, delta=1):
        self.graph = graph
        self.delta = delta

    def step(self):
        mega = self.graph.get_mega_urns()  # num_nodes x num_colors)
        # Compute sums of mega balls in each mega urn
        row_sums = cp.sum(mega, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # if mega urn is zero, avoid div by zero
        # Create probability matrix holding color probabilities for each node's mega urn
        probs = mega / row_sums  #(num_nodes x num_colors)

        # Vectorized sampling
        # Generate random vals for each node
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
        self.graph.node_urns[rows, draws] += self.delta
