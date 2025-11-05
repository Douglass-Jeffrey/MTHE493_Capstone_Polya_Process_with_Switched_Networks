from .node import Node
import matplotlib.pyplot as plt  # for plotting graphs
import numpy as np  # will need numpy since numpy arrays are faster than dicts apparently

class Graph:    
    def __init__(self):
        # dictionary of node ids to node instances
        self._nodes = {}  # id -> Node
    
    @property
    def nodes(self):
        return self._nodes.keys()
    
    def add_node(self, node_id, value=None):
        if node_id not in self._nodes:
            self._nodes[node_id] = Node(node_id, value)

    def add_edge(self, from_node, to_node, directed=False): 
        if from_node in self._nodes and to_node in self._nodes:
            self._nodes[from_node].add_edge(to_node)
            if not directed:
                self._nodes[to_node].add_edge(from_node)
    
    def get_node(self, node_id):
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph")
        return self._nodes[node_id]
    
    def get_nodes(self):
        return (self._nodes)

    def get_neighbours(self, node_id):
        return self.get_node(node_id).get_edges()


    