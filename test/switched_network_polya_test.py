import sys
import os
import random
import time
# Add the project's src directory to sys.path so we can import the package during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from capstone.process_animator import ProcessVisualizer
from capstone.switch_network_process import Switched_Network
from capstone.polya_process import Polya
from capstone.graphing import Graph

if __name__ == "__main__":

    start = time.time()

    # Create a graph instance
    graph = Graph()

    # Create new polya process with the graph
    polya = Polya(delta=10, starting_graph=graph)
    
    # Create new switched network process with the graph, and custom switching rule
    switch_function = lambda: random.randint(0, 5) > 4
    sw = Switched_Network(switch_rule=switch_function, starting_graph=graph)

    # Create new visualizer for sw
    #vi = ProcessVisualizer(sw) 
    # Create new visualizer for polya
    iv = ProcessVisualizer(sw)
    vi = ProcessVisualizer(polya)

    print("Begin generating switched network...")
    vi.animate(
        step_function=lambda: sw.step(),
        steps=50,
        interval=10,
        title="50 Step Switched Network with 1/5 Probability of Link",
        save=True,
        output_path="test_4_switched_network_50_prob_1_in_5.mp4",
    )
    print("Done generating switched network!")

    print("Begin setting node values...")
    for node_id, node in sw.graph.nodes.items():
        node.urn.add_item("Blue", 1)
        node.urn.add_item("Red", 1)
        node.mega_urn = node.urn.contents.copy()
        node.urn.last_drawn_item = random.choice(["Blue", "Red"])  # random initial color
    print("Done setting node values!")

    print("Begin running polya process on switched network...")
    vi.animate(
        step_function=lambda: polya.step(),
        steps=50000,
        interval=2,
        title="10 Delta 50000 Step Polya Process on Switched Network with Urns Initialized at 1:1",
        save=True,
        output_path="test_4_polya_50000_delta_10_init_1_1.mp4",
    )
    print("Done running polya process on switched network!")

    for node_id, node in polya.graph.nodes.items():
        print(node_id, node.urn.contents) 
        print(node_id, node.mega_urn) 

    delta = time.time() - start
    print(f'Took {delta:.2f} seconds')


    #print("dictionary: ", sw.get_graph().get_nodes())