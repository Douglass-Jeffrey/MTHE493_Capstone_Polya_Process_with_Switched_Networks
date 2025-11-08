from .graphing import Graph
from .urn import Urn
import copy
import random

DELTA = 1

class Polya():
    def __init__(self, delta=None, starting_graph=None):
        if starting_graph is not None:
            self._graph = starting_graph
        else:
            self._graph = Graph()

        if delta is not None:
            self._delta = delta
        else:
            self._delta = DELTA

    @property
    def graph(self):
        return self._graph

    @property
    def delta(self):
        return self._delta

    def mega_urn_for_node(self, node_id, node):
        mega_urn = Urn()
        neighbours = self.graph.get_neighbours(node_id)
        for nid in neighbours:
            neighbour_node = self.graph.get_node(nid)
            neighbout_urn = neighbour_node.urn
            contents = neighbout_urn.contents
            for key in contents.keys():
                mega_urn.add_item(key, contents[key])
        return mega_urn

    def update_urns(self, node_id, item_id):
        node = self.graph.get_node(node_id)
        node.urn.add_item(item_id, self.delta)
        neighbours = self.graph.get_neighbours(node_id)
        for nid in neighbours:
            neighbour_node = self.graph.get_node(nid)
            neighbour_urn = neighbour_node.urn
            neighbour_urn.add_item(item_id, self.delta) 

    def step(self):
        original_graph = copy.deepcopy(self.graph)
        for node_id, node in original_graph.nodes.items():
            if node.num_edges == 0:
                continue
            else:
                mega_urn = self.mega_urn_for_node(node_id, node)
                choice = mega_urn.choose_random_item()
                self.update_urns(node_id, choice)

