import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.polya_process import Polya
from capstone.urn import urn

if __name__ == "__main__":
    poly_graph = Polya()

    # add four nodes labeled A..D with identical starting urns
    for label in ["A", "B", "C", "D"]:
        poly_graph.add_node(node_id=label)
        poly_graph.get_node(label).get_urn().add_item("Black", 2)
        poly_graph.get_node(label).get_urn().add_item("Red", 2)

    # connect them in a cycle so each node has degree 2: A-B-C-D-A
    poly_graph.add_edge("A", "B")
    poly_graph.add_edge("B", "C")
    poly_graph.add_edge("C", "D")
    poly_graph.add_edge("D", "A")

    for node in poly_graph.get_nodes():
        urn_node = poly_graph.get_node(node).get_urn()
        contents = urn_node.get_contents()
        print(f"Node {node}: {contents}")
        
    poly_graph.run_polya()

    for node in poly_graph.get_nodes():
        node_urn = poly_graph.get_node(node).get_urn()
        print(f"Node {node}: {node_urn.contents}")
