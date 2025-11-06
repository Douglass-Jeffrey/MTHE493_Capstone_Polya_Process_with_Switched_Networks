from capstone.graphing import Graph

class Switched_Network:
    def __init__(self, switch_rule, starting_graph=None):
        self.switch_rule = switch_rule
        self.graph = starting_graph if starting_graph is not None else Graph()

    def step(self, function=None):
        """Perform one switched network update."""
        rule = function if function is not None else self.switch_rule
        if rule():
            new_node_id = self.graph.add_node()
            for existing_node_id in self.graph.get_nodes():
                if existing_node_id != new_node_id:
                    self.graph.add_edge(new_node_id, existing_node_id)
        else:
            self.graph.add_node()

    def get_graph(self):
        return self.graph
            

        


