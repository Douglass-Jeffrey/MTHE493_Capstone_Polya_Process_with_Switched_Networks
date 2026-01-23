import sys
import os
import random
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

CUPY_AVAILABLE = False

from capstone_optimized.core import Graph, Barabasi_Albert_Growth, Polya_Process, Polya_Process_Animator
# ignore above 

# create graph
G = Graph(num_nodes=100, num_colors=2, use_gpu=False)

x = Graph()

#grow the graph
B_A_growth = Barabasi_Albert_Growth(G, m=1)
B_A_growth.grow()
B_A_growth.finalize_growth()

#run the polya process
polya = Polya_Process(G, delta=1)
#polya.run(steps=500)

# create animator 
anim = Polya_Process_Animator(G, polya, node_size=80, interval=50)
print('Animator created, nodes=', G.num_nodes, 'edges=', G.num_edges)

# animate the process
anim.animate(steps=200, save_path="polya_demo.mp4")
print('Update OK')
