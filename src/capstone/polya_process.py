from .graphing import Graph
from .urn import Urn
import random

DELTA = 1

class Polya:
    def __init__(self, delta=None, starting_graph=None):
        self._graph = starting_graph if starting_graph is not None else Graph()
        self._delta = delta if delta is not None else DELTA

    @property
    def graph(self):
        return self._graph

    @property
    def delta(self):
        return self._delta
    """
    def mega_urn_for_node(self, node_id, node):
        mega_urn = Urn()        
        # add the nodes urns contents
        for color, qty in node.urn.contents.items():
            mega_urn.add_item(color, qty)
        # add neighbor urn contents
        for neighbor_id in self.graph.get_neighbours(node_id):
            neighbor_node = self.graph.get_node(neighbor_id)
            for color, qty in neighbor_node.urn.contents.items():
                mega_urn.add_item(color, qty)
        return mega_urn
    """
    def update_urn(self, node_id, item_id):
        node = self.graph.get_node(node_id)
        node.urn.add_item(item_id, self.delta)
        node.mega_urn[item_id] += self.delta

        for nbr_id in self.graph.get_neighbours(node_id):
            nbr = self.graph.get_node(nbr_id)
            nbr.mega_urn[item_id] += self.delta

        node.urn.last_drawn_item = item_id

    def step(self):
        choices = {}
        
        for node_id, node in self.graph.nodes.items():
            if not node.mega_urn:
                continue
            
            # draw from cached mega urn
            item = random.choices(
                population=list(node.mega_urn.keys()),
                weights=list(node.mega_urn.values()),
                k=1
            )[0]

            choices[node_id] = item

        # apply updates
        for node_id, item in choices.items():
            self.update_urn(node_id, item)