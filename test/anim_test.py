import sys
import os
import random
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.switch_network import Switched_Network
from capstone.graphing import Graph
from capstone.network_animator import NetworkVisualizer

if __name__ == "__main__":
    
    #switch_function = lambda: True
    switch_function = lambda: random.randint(0, 100) > 99

    # create switched network with empty graph
    sw = Switched_Network(switch_rule=switch_function)
    vi = NetworkVisualizer(sw)


    # Animate your existing step function
    vi.animate(
        step_function=lambda: sw.step(),
        steps=1000,
        interval=10,
        title="Switched Network Growth",
        layout="random",
        save=True,
        output_path="1000_nodes_1_percent_connection_rate_switched_network.mp4",
    )
