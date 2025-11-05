import sys
import os
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.graphing import Graph
from capstone.urn import urn

if __name__ == "__main__":

    # Create a graph instance
    graph = Graph()
    # Add nodes
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C") 
    graph.add_node("D")
    # Add edges 
    graph.add_edge("A", "B")
    graph.add_edge("A", "C", directed=True)
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")

    # Display neighbours
    for node_id in graph.get_nodes():
        neighbours = graph.get_neighbours(node_id)
        print(f"Node {node_id} has neighbours: {neighbours}")
    
    graph.get_node("A").get_urn().add_item("Black", 5)
    graph.get_node("A").get_urn().add_item("Red", 4)

    contents = graph.get_node("A").get_urn().get_contents()
    print(f"Contents of urn at node A: {contents}")