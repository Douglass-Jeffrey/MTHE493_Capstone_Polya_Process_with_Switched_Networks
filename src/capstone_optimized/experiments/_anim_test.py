import sys
import os
import random
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

CUPY_AVAILABLE = False

from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process, Polya_Process_Animator

G = Graph(num_nodes=50, num_colors=2, use_gpu=False)
grow = Barabasi_Albert_Growth(G, m=2)
grow.grow()
grow.finalize_growth()
polya = Polya_Process(G, delta=1)
anim = Polya_Process_Animator(G, polya, node_size=80, interval=50)
print('Animator created, nodes=', G.num_nodes, 'edges=', G.num_edges)
# do one update to ensure no error
anim._update(0)
print('Update OK')
