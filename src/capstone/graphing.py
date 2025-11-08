from .node import Node
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Graph:
    def __init__(self):
        # dictionary of node ids to node instances
        self._nodes = {}  # id -> Node
        self._num_nodes = 0
        self._num_edges = 0

    # --------------------
    # Property definitions
    # --------------------
    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        if not isinstance(value, dict):
            raise TypeError("nodes must be a dictionary mapping ids to Node objects")
        self._nodes = value

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("num_nodes must be a non-negative integer")
        self._num_nodes = value

    @property
    def num_edges(self):
        return self._num_edges

    @num_edges.setter
    def num_edges(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("num_edges must be a non-negative integer")
        self._num_edges = value

    # --------------------
    # Graph operations
    # --------------------
    def add_node(self, node_id=None):
        if node_id is None:
            node_id = self.num_nodes + 1

        # Create and add node
        self.nodes[node_id] = Node(node_id)
        self.num_nodes = self.num_nodes + 1
        return node_id

    def add_edge(self, from_node, to_node, directed=False):
        if from_node in self.nodes and to_node in self.nodes:
            self.nodes[from_node].add_edge(to_node)
            if not directed:
                self.nodes[to_node].add_edge(from_node)
            self.num_edges = self.num_edges + 1
        else:
            raise KeyError("Both nodes must exist in the graph before adding an edge.")

    def get_node(self, node_id):
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in graph")
        return self.nodes[node_id]

    def get_neighbours(self, node_id):
        return self.get_node(node_id).edges
