from .node import Node
import matplotlib.pyplot as plt  # for plotting graphs
import networkx as nx  # for graph visualization
import numpy as np  # will need numpy since numpy arrays are faster than dicts apparently

class Graph:    
    def __init__(self):
        # dictionary of node ids to node instances
        self._nodes = {}  # id -> Node
        self._num_nodes = 0
    
    @property
    def nodes(self):
        return self._nodes.keys()
    
    """DJEFFREY TODO:
    possible problem, someone defines a node of id 100 and then adds nodes without specifying id, when it gets to 100 there will be a conflict"""
    def add_node(self, node_id=None, value=None):
        if node_id is None:
            node_id = self._num_nodes + 1
        self._nodes[node_id] = Node(node_id, value)
        self._num_nodes = self._num_nodes + 1
        #print("added node with nodeid:", node_id)
        return node_id

    def add_edge(self, from_node, to_node, directed=False):
        if from_node in self._nodes and to_node in self._nodes:
            self.get_node(from_node).add_edge(to_node)
            if not directed:
                self.get_node(to_node).add_edge(from_node)
    
    def get_node(self, node_id):
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph")
        return self._nodes[node_id]
    
    def get_num_nodes(self):
        return self._num_nodes
    
    def get_nodes(self):
        return (self._nodes)

    def get_neighbours(self, node_id):
        return self.get_node(node_id).get_edges()

    def plot_graph(graph, title="Network Graph", layout="spring"):
        """
        Plot a Graph object using networkx + matplotlib.
        Assumes each node object has a get_edges() method.
        """
        G = nx.DiGraph()  # directed graph (can show one-way edges)

        # Add nodes and edges from your custom Graph
        for node_id, node in graph.get_nodes().items():
            G.add_node(node_id)
            for neighbour in node.get_edges():
                G.add_edge(node_id, neighbour)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G)

        # Draw the graph
        plt.figure(figsize=(8, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=600,
            node_color="lightblue",
            font_weight="bold",
            arrows=True,
            arrowsize=15
        )
        plt.title(title)
        plt.show()


    