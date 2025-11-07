from .urn import urn

class Node:
    """A node in the graph containing an urn and connected edges."""
    
    def __init__(self, node_id, value=None):
        self.id = node_id
        self._urn = urn()
        self._edges = set()
        self.value = value
        self.num_edges = 0
    
    def get_urn(self):
        return self._urn
    
    def get_edges(self):
        return set(self._edges)
    
    def add_edge(self, other_node_id):
        self._edges.add(other_node_id)
        self.num_edges += 1
    
    def remove_edge(self, other_node_id):
        self._edges.remove(other_node_id)
        self.num_edges -= 1