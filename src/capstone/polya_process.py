from .graph import Graph
from.node import Node
from .urn import urn

""" standard polya process: at each step update every node in the graph at once based on super urn"""
class Polya_Process:
    def __init__ (self, starting_graph=None):
        if starting_graph is not None:
            self.graph = starting_graph
        else:
            self.graph = Graph()
    # DJEFFREY TODO: should we keep the graph sorted based on # of edges?
    # this will reduce computation complexity for graphs with high number of nodes 
    # could also just swap to numpy arrays or some other effecient data stucture
    def step(self):
        
        for node in graph.get_nodes():
            for neighbour_id in graph.get_neighbours(node):



