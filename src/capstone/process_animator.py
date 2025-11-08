import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

class ProcessVisualizer:

    def __init__(self, process):
        self.process = process              
        self.graph = process.graph    
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.pos = None                      

    # for static graph plotting
    def plot_graph(self, title="Process Graph", layout="spring"):
        G = nx.DiGraph()
        for node_id, node in self.graph.nodes.items():
            G.add_node(node_id)
            for neighbour in node.edges:
                G.add_edge(node_id, neighbour)

        n = G.number_of_nodes()
        m = G.number_of_edges()

        if n > 2000:
            pos = nx.random_layout(G, seed=42)
        elif layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=1 / ((n ** 0.5) * 3), iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        node_size = max(5, 100000 / n)
        edge_width = max(0.1, 1000 / n)
        alpha = 0.4 if n < 5000 else 0.1

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="dodgerblue", alpha=alpha)
        if m < 200000:
            nx.draw_networkx_edges(G, pos, width=edge_width, alpha=alpha * 0.6, arrows=False)

        plt.title(f"{title}\nNodes: {n:,}, Edges: {m:,}", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # for animated graph plotting
    #DJEFFREY TODO: should change args to allow step function to be passed in with args
    # would stop us from having to predefine step function for a process first in test before passing to animate
    # would also allow us to change step function on the fly
    def animate(
        self,
        step_function,
        steps=100,
        interval=200,
        layout="spring",
        title="Graph Animation",
        save=False,
        output_path="animation.mp4",
        ):

        self.ax.clear()
        self.ax.axis("off")
        self.ax.set_title(title)

        # node positions set as an empty dictionary to be populated over time
        self.pos = {}

        def update(frame):
            # apply user-defined step function
            step_function()

            # build networkx graph from current state
            G = nx.DiGraph()
            for node_id, node in self.graph.nodes.items():
                G.add_node(node_id)
                for neighbour in node.edges:
                    G.add_edge(node_id, neighbour)

            n = G.number_of_nodes()
            m = G.number_of_edges()

            # skip drawing if graph is empty
            if n == 0:
                return []

            # ensure all nodes have an initial position
            import numpy as np
            for node in G.nodes:
                if node not in self.pos:
                    self.pos[node] = np.random.rand(2)

            # refine layout for smooth animation
            if layout == "spring" and n > 1:
                self.pos = nx.spring_layout(G, pos=self.pos, fixed=self.pos.keys(), iterations=10, seed=42)
            elif layout == "circular":
                self.pos = nx.circular_layout(G)
            elif layout == "random":
                for node in G.nodes:
                    if node not in self.pos:
                        self.pos[node] = np.random.rand(2)

            # clear axis before drawing
            self.ax.clear()
            self.ax.axis("off")

            # scale node size and edge width based on total number of steps 
            node_size = max(5, 10000 / steps)
            edge_width = max(0.1, 100 / steps)
            alpha = 0.4 if n < 5000 else 0.1

            # draw nodes
            nx.draw_networkx_nodes(G, self.pos, node_size=node_size, node_color="dodgerblue", alpha=alpha, ax=self.ax)
            # draw edges if not too many
            if m < 200000:
                nx.draw_networkx_edges(G, self.pos, width=edge_width, alpha=alpha * 0.6, arrows=False, ax=self.ax)

            self.ax.set_title(f"{title} — Step {frame+1}\nNodes: {n:,}, Edges: {m:,}")

            return []

        # create animation
        anim = FuncAnimation(self.fig, update, frames=steps, interval=interval, repeat=False)

        if save: # note need to have ffmpeg installed for saving, pillow doesnt work as far as ive tried
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=1000 // interval, bitrate=1800)
                anim.save(output_path, writer=writer)
            except Exception as e:
                print("FFmpeg not available, saving as GIF instead:", e)
            anim.save(output_path.replace(".mp4", ".gif"), writer="pillow", fps=1000 // interval)
        else:
            plt.show()
