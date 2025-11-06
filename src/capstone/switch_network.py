from .graphing import Graph
class Switched_Network:
    def __init__ (self, starting_graph=None):
        #instantiate a graph which we will grow with the switched network process
        if starting_graph is not None:
            self.graph = starting_graph
        else:   
            self.graph = Graph()

    def switched_network_step(self, function):
        if function() == True:
            new_node_id = self.graph.add_node()
            gen = (existing_node_id for existing_node_id in self.graph.get_nodes() if existing_node_id != new_node_id)
            for existing_node_id in gen:
                #print("adding edge between", new_node_id, "and", existing_node_id)
                self.graph.add_edge(new_node_id, existing_node_id)
        else:
            self.graph.add_node()

    def get_graph(self):
        return self.graph
            

        


