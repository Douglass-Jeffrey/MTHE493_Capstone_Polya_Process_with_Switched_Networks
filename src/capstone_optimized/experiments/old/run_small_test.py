import sys
import os

# Ensure top-level 'src' is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from capstone_optimized.core import Graph, PolyaProcess, SwitchNetworkGrowth

def main():
    N = 1000
    graph = Graph(num_nodes=N, num_colors=2, use_gpu=True)

    growth = SwitchNetworkGrowth(graph, connection_prob=0.3)
    for _ in range(5):
        growth.grow_batch(batch_size=100)

    growth.finalize_growth()

    polya = PolyaProcess(graph, delta=1)
    for _ in range(2):
        polya.step()

    print('SMALL_TEST_DONE', 'nodes=', graph.num_nodes, 'edges=', graph.num_edges)

if __name__ == '__main__':
    main()
