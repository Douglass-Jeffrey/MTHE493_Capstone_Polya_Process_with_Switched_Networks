import sys
import os
import random
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.switch_network import Switched_Network
from capstone.graphing import Graph

if __name__ == "__main__":

    # Create a graph instance
    graph = Graph()
    # Add nodes
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3) 
    graph.add_node(4)
    # Add edges 
    graph.add_edge(1, 2)
    graph.add_edge(1, 3, directed=True)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    sw = Switched_Network(graph)

    #switch_function = lambda: True
    switch_function = lambda: random.randint(0, 50) > 49

    for i in range(500):
        sw.switched_network_step(switch_function)

    # Display neighbours
    #for node_id in sw.get_graph().get_nodes():
    #    neighbours = graph.get_neighbours(node_id)
    #    print(f"Node {node_id} has neighbours: {neighbours}")

    #sw.get_graph().plot_graph(title="Switched Network after 1000 steps", layout="spring")
    #sw.get_graph().plot_graph(title="Switched Network after 1000 steps", layout="circular")
    sw.get_graph().plot_graph(title="Switched Network after 1000 steps", layout="random")
    #sw.get_graph().plot_graph(title="Switched Network after 1000 steps", layout="kamada_kawai")


    #print("dictionary: ", sw.get_graph().get_nodes())