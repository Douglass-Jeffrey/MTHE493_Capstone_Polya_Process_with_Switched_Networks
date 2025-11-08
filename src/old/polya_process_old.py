from .graphing import Graph
from .urn import urn
import copy
import random

DELTA = 10

class Polya(Graph):
    def __init__(self):
        super().__init__()
        self.delta = DELTA

    def mega_urn_for_node(self, node_id):
        mega_urn = urn()
        neighbours = self.get_neighbours(node_id)
        for nid in neighbours:
            neighbour_node = self.get_node(nid)
            urn_node = neighbour_node.get_urn()
            contents = urn_node.get_contents()
            for key in contents.keys():
                mega_urn.add_item(key, contents[key])

        ##print(f"{mega_urn.contents}")
        return mega_urn

    def update_urns(self, node_id, item_id):
        node = self.get_node(node_id)
        node.get_urn().add_item(item_id, self.delta)
        neighbours = self.get_neighbours(node_id)
        for nid in neighbours:
            neighbour_node = self.get_node(nid)
            neighbour_urn = neighbour_node.get_urn()
            neighbour_urn.add_item(item_id, self.delta) 

    def run_polya(self):
        original_graph = copy.deepcopy(self)
        for node in original_graph.get_nodes():
            if self.get_node(node).num_edges == 0:
                continue
            else:
                mega_urn = original_graph.mega_urn_for_node(node)
                print(mega_urn.contents)
                choice = mega_urn.choose_random_ball()
                #print(f"{node} : {choice}")
                self.update_urns(node, choice)



