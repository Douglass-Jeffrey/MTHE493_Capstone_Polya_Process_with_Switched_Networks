from .urn import Urn

class Node:
    """A node in the graph containing an urn and connected edges."""
    
    def __init__(self, node_id):
        self._node_id = node_id
        self._urn = Urn()
        self._edges = set()
        self._num_edges = 0

    # --------------------
    # Property definitions
    # --------------------
    @property
    def node_id(self):
        """Unique ID of the node (read-only)."""
        return self._node_id

    @property
    def urn(self):
        """The Urn object associated with this node."""
        return self._urn

    @urn.setter
    def urn(self, value):
        if not isinstance(value, Urn):
            raise TypeError("urn must be an instance of Urn")
        self._urn = value

    @property
    def edges(self):
        """A set of connected node IDs."""
        return self._edges

    @edges.setter
    def edges(self, value):
        if not isinstance(value, set):
            raise TypeError("edges must be a set of node IDs")
        self._edges = value

    @property
    def num_edges(self):
        """Number of connected edges."""
        return self._num_edges

    @num_edges.setter
    def num_edges(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("num_edges must be a non-negative integer")
        self._num_edges = value

    # --------------------
    # Node operations
    # --------------------
    def add_edge(self, other_node_id):
        """Add an edge to another node."""
        if other_node_id not in self._edges:
            self._edges.add(other_node_id)
            self._num_edges += 1

    def remove_edge(self, other_node_id):
        """Remove an existing edge to another node."""
        if other_node_id in self._edges:
            self._edges.remove(other_node_id)
            self._num_edges -= 1
