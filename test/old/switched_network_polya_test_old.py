import sys
import os
import random
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.network_animator import NetworkVisualizer
from capstone.switch_network import Switched_Network
from capstone.polya_process import Polya
from capstone.graphing import Graph

if __name__ == "__main__":

    # Create a graph instance
    polya = Polya()
        
    switch_function = lambda: random.randint(0, 1) > 0
    sw = Switched_Network(switch_rule=switch_function, starting_graph=polya)
    vi = NetworkVisualizer(sw) 

    for i in range(100):
        sw.step()

    #vi.plot_graph("random layout", layout="random")

    for label in sw.get_graph().get_nodes():
        sw.get_graph().get_node(label).get_urn().add_item("Black", 1)
        sw.get_graph().get_node(label).get_urn().add_item("Red", 1)

    nodes = polya.get_nodes().values()
    for node in nodes:
        print(node.get_urn().contents)

    for i in range(100):
        polya.run_polya()
    print("done polya")
    nodes = polya.get_nodes().values()
    for node in nodes:
        print(node.get_urn().contents) 



    #print("dictionary: ", sw.get_graph().get_nodes())