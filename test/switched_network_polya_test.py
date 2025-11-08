import sys
import os
import random
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.process_animator import ProcessVisualizer
from capstone.switch_network_process import Switched_Network
from capstone.polya_process import Polya
from capstone.graphing import Graph

if __name__ == "__main__":

    # Create a graph instance
    graph = Graph()

    # Create new polya process with the graph
    polya = Polya(delta=10, starting_graph=graph)
    
    # Create new switched network process with the graph, and custom switching rule
    switch_function = lambda: random.randint(0, 75) > 74
    sw = Switched_Network(switch_rule=switch_function, starting_graph=graph)

    # Create new visualizer for sw
    #vi = ProcessVisualizer(sw) 
    # Create new visualizer for polya
    iv = ProcessVisualizer(sw)
    vi = ProcessVisualizer(polya)

    print("Begin generating switched network...")
    vi.animate(
        step_function=lambda: sw.step(),
        steps=500,
        interval=100,
        title="500 Step Switched Network with 1/75 Probability of Link",
        save=True,
        output_path="test_2_switched_network_500_prob_1_in_75.mp4",
    )
    print("Done generating switched network!")

    print("Begin setting node values...")
    for node_id, node in sw.graph.nodes.items():
        node.urn.add_item("Blue", 10)
        node.urn.add_item("Red", 1)
        node.urn.last_drawn_item = random.choice(["Blue", "Red"])  # random initial color
    print("Done setting node values!")

    print("Begin running polya process on switched network...")
    vi.animate(
        step_function=lambda: polya.step(),
        steps=1000,
        interval=100,
        title="10 Delta 1000 Step Polya Process on Switched Network with Urns Initialized at 1:10",
        save=True,
        output_path="test_2_polya_1000_delta_10_init_1_10.mp4",
    )
    print("Done running polya process on switched network!")

    for node_id, node in polya.graph.nodes.items():
        print(node_id, node.urn.contents) 



    #print("dictionary: ", sw.get_graph().get_nodes())