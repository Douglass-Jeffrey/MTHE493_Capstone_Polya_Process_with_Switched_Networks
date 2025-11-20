from .urn import Urn
from collections import Counter

class Node:
    
    def __init__(self, node_id):
        self._node_id = node_id
        self._urn = Urn()
        self._edges = set()
        self._num_edges = 0
        self._mega_urn = Counter()

    @property
    def node_id(self):
        return self._node_id

    @property
    def urn(self):
        return self._urn

    @property
    def mega_urn(self):
        return self._mega_urn

    @mega_urn.setter
    def mega_urn(self, value):
        if not isinstance(value, dict):
            raise TypeError("mega_urn must be a dictionary mapping item_id -> quantity")
        for qty in value.values():
            if not isinstance(qty, int) or qty < 0:
                raise ValueError("All quantities in mega_urn must be non-negative integers")
        self._mega_urn = Counter(value)

    @urn.setter
    def urn(self, value):
        if not isinstance(value, Urn):
            raise TypeError("urn must be an instance of Urn")
        self._urn = value

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        if not isinstance(value, set):
            raise TypeError("edges must be a set of node IDs")
        self._edges = value

    @property
    def num_edges(self):
        return self._num_edges

    @num_edges.setter
    def num_edges(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("num_edges must be a non-negative integer")
        self._num_edges = value


    def add_edge(self, other_node_id):
        if other_node_id not in self._edges:
            self._edges.add(other_node_id)
            self._num_edges += 1

    def remove_edge(self, other_node_id):
        if other_node_id in self._edges:
            self._edges.remove(other_node_id)
            self._num_edges -= 1
