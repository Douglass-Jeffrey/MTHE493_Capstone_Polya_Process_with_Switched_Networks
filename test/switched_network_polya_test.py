import sys
import os
import random
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.process_animator import ProcessVisualizer
from capstone.switch_network_process import Switched_Network
from capstone.polya_process import Polya
from capstone.graphing import Graph

if __name__ == "__main__":

    # Create a graph instance
    graph = Graph()

    # Create new polya process with the graph
    polya = Polya(delta=3, starting_graph=graph)
    
    # Create new switched network process with the graph, and custom switching rule
    switch_function = lambda: random.randint(0, 1) > 0
    sw = Switched_Network(switch_rule=switch_function, starting_graph=graph)

    # Create new visualizer for sw
    vi = ProcessVisualizer(sw) 
    # Create new visualizer for polya
    vi = ProcessVisualizer(polya)

    for i in range(100):
        sw.step()

    #vi.plot_graph("random layout", layout="random")

    for node_id, node in sw.graph.nodes.items():
        print(node)

    for node_id, node in sw.graph.nodes.items():
        node.urn.add_item("Black", 10)
        node.urn.add_item("Red", 10)

    for node_id, node in sw.graph.nodes.items():
        print(node)

    for i in range(100):
        polya.step()

    print("done polya")

    for node_id, node in polya.graph.nodes.items():
        print(node.urn.contents) 



    #print("dictionary: ", sw.get_graph().get_nodes())