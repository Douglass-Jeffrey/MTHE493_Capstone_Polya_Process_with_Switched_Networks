from ..cupy_fallback import cp, CUPY_AVAILABLE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False


class Polya_Process_Animator:
    """Animate a Polya process running on a Graph.

    - Nodes are coloured `red` or `blue` depending on whichever colour
      they have more of in their urns at each step.
    - Edges are drawn and remain visible behind the nodes.

    Parameters
    - graph: instance of `Graph` from `graph_x.py` (must have `adj_matrix` and
      `node_urns`).
    - polya: instance of `Polya_Process` (must implement `step()`).
    - node_size: size for node markers.
    - interval: ms between frames.
    - figsize: tuple for figure size.
    """

    def __init__(self, graph, polya, node_size=50, interval=500, figsize=(8, 6), verbose=False, print_every=1):
        self.graph = graph
        self.polya = polya
        self.node_size = node_size
        self.interval = interval
        self.figsize = figsize
        # printing control
        self.verbose = bool(verbose)
        self.print_every = int(print_every) if print_every and int(print_every) > 0 else 1

        # colour map for majority index -> colour
        self._colour_map = {0: 'red', 1: 'blue'}

        # Prepare node positions and edges for drawing
        self._prepare_graph_visual()

        # Matplotlib state
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._init_plot_elements()

        self._step_count = 0

    def _prepare_graph_visual(self):
        # Build edge list from adjacency matrix. Support cupy/cupyx and scipy.
        adj = self.graph.adj_matrix
        if adj is None:
            # no edges
            self.edges = []
        else:
            coo = adj.tocoo()
            rows = coo.row
            cols = coo.col
            # if cupy arrays, convert to numpy
            try:
                if hasattr(rows, 'get'):
                    rows = rows.get()
                if hasattr(cols, 'get'):
                    cols = cols.get()
            except Exception:
                # last resort: try cp.asnumpy
                try:
                    if CUPY_AVAILABLE and hasattr(cp, 'asnumpy'):
                        rows = cp.asnumpy(rows)
                        cols = cp.asnumpy(cols)
                except Exception:
                    pass

            # collect unique undirected edges for visualization
            edge_set = set()
            for r, c in zip(rows, cols):
                if r == c:
                    continue
                a, b = int(r), int(c)
                if a <= b:
                    edge_set.add((a, b))
                else:
                    edge_set.add((b, a))
            self.edges = list(edge_set)

        # compute node positions: use spring layout for connected nodes and
        # place isolated nodes on a surrounding circle to avoid overlap in
        # the centre (networkx tends to place isolated nodes near origin).
        n = int(self.graph.num_nodes)
        # default positions
        self.pos = np.zeros((n, 2), dtype=float)

        if getattr(self, 'edges', None) and len(self.edges) > 0 and _HAS_NX:
            # compute degrees and split connected vs isolated nodes
            deg = [0] * n
            for a, b in self.edges:
                deg[a] += 1
                deg[b] += 1
            connected = [i for i in range(n) if deg[i] > 0]
            isolated = [i for i in range(n) if deg[i] == 0]

            # build subgraph for connected nodes only (keeps isolated nodes out of spring)
            sub = nx.Graph()
            sub.add_nodes_from(connected)
            sub.add_edges_from([(a, b) for (a, b) in self.edges if a in connected and b in connected])

            if len(connected) > 0:
                # choose a scale to spread nodes: scale increases with sqrt of size
                scale = max(1.0, float(np.sqrt(len(connected))))
                pos_sub = nx.spring_layout(sub, seed=1, scale=scale)
                for node, p in pos_sub.items():
                    self.pos[int(node), :] = p

            # place isolated nodes evenly on a circle outside the main layout
            if len(isolated) > 0:
                radius = max(2.0, float(np.sqrt(n))) * 1.2
                m = len(isolated)
                angles = 2 * np.pi * np.arange(m) / max(1, m)
                for idx, node in enumerate(isolated):
                    self.pos[int(node), 0] = radius * np.cos(angles[idx])
                    self.pos[int(node), 1] = radius * np.sin(angles[idx])
        else:
            # circular layout fallback for all nodes
            theta = 2 * np.pi * np.arange(n) / max(1, n)
            self.pos = np.column_stack([np.cos(theta), np.sin(theta)])

    def _init_plot_elements(self):
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # draw edges as a LineCollection so they're efficient
        if getattr(self, 'edges', None):
            segments = []
            for a, b in self.edges:
                segments.append((self.pos[a], self.pos[b]))
            # make edges darker and slightly thicker so they remain visible
            self.edge_collection = LineCollection(segments, colors='#333333', linewidths=0.8, alpha=0.9)
            self.edge_collection.set_zorder(1)
            self.ax.add_collection(self.edge_collection)
        else:
            self.edge_collection = None

        # initial node colours by majority
        colours = self._majority_colours()
        xs = self.pos[:, 0]
        ys = self.pos[:, 1]
        self.node_scatter = self.ax.scatter(xs, ys, s=self.node_size, c=colours, edgecolors='black', linewidths=0.4)
        self.node_scatter.set_zorder(3)
        self.title = self.ax.text(0.5, 1.03, 'Barabasi-Albert Networked Polya process step 0', transform=self.ax.transAxes, ha='center')
        # ensure axes cover all nodes and edges (with padding) so long edges are visible
        try:
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            ymin, ymax = float(np.min(ys)), float(np.max(ys))
            dx = max(1e-6, xmax - xmin)
            dy = max(1e-6, ymax - ymin)
            pad = 0.15 * max(dx, dy)
            self.ax.set_xlim(xmin - pad, xmax + pad)
            self.ax.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            # fall back to autoscale
            try:
                self.ax.relim()
                self.ax.autoscale_view()
            except Exception:
                pass

    def _get_node_urns_numpy(self):
        urns = self.graph.node_urns
        try:
            if hasattr(urns, 'get'):
                urns = urns.get()
        except Exception:
            try:
                if CUPY_AVAILABLE and hasattr(cp, 'asnumpy'):
                    urns = cp.asnumpy(urns)
            except Exception:
                pass
        return np.asarray(urns)

    def _print_graph_state(self):
        # Print a readable summary of the graph to console.
        urns = self._get_node_urns_numpy()
        n = int(self.graph.num_nodes)
        ecount = int(self.graph.num_edges) if hasattr(self.graph, 'num_edges') else (len(self.edges) if getattr(self, 'edges', None) is not None else 0)
        print(f'--- Polya step {self._step_count} | nodes={n} edges={ecount} ---')

        # Print urns: if small graph, print full matrix, otherwise print summary
        if n <= 200:
            # show full urn matrix
            with np.printoptions(edgeitems=10, linewidth=200, threshold=1000):
                print('Node urns (rows=node, cols=colours):')
                print(urns)
        else:
            # print majority counts and a small sample
            idx = np.argmax(urns, axis=1)
            reds = int((idx == 0).sum())
            blues = int((idx == 1).sum()) if urns.shape[1] > 1 else 0
            print(f'Majority counts: red={reds} blue={blues}')
            sample_n = min(20, n)
            print(f'First {sample_n} urns:')
            print(urns[:sample_n])

        # Print a small sample of edges (if available)
        if getattr(self, 'edges', None):
            print(f'Edges (showing up to 50 unique undirected edges): total_unique={len(self.edges)}')
            for a, b in self.edges[:5000]:
                print(f'{a} - {b}')
        print('--------------------------------------------')

    def _majority_colours(self):
        urns = self._get_node_urns_numpy()
        # argmax along colours axis
        idx = np.argmax(urns, axis=1)
        # map indices to colour names
        return [self._colour_map.get(int(i), 'gray') for i in idx]

    def _update(self, frame):
        # advance polya process by one step
        self.polya.step()
        self._step_count += 1

        # update node colours
        colours = self._majority_colours()
        self.node_scatter.set_facecolor(colours)
        self.node_scatter.set_edgecolor('k')
        self.title.set_text(f'Barabasi-Albert Networked Polya process step {self._step_count}')
        # optional console output
        if self.verbose and (self._step_count % self.print_every) == 0:
            try:
                self._print_graph_state()
            except Exception as e:
                print('Warning: failed to print graph state:', e)
        return (self.node_scatter, self.title)

    def animate(self, steps=100, save_path=None, dpi=150):
        """Run the animation for `steps` steps. If `save_path` is provided,
        attempt to save the animation to that path (mp4 or gif). Otherwise
        display the animation window.
        """
        anim = FuncAnimation(self.fig, self._update, frames=range(steps), interval=self.interval, blit=False)

        if save_path:
            try:
                if save_path.lower().endswith('.mp4'):
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=max(1, int(1000 // self.interval)))
                    anim.save(save_path, writer=writer, dpi=dpi)
                else:
                    # fallback to gif
                    from matplotlib.animation import PillowWriter
                    fps = max(1, int(1000 // self.interval))
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path, writer=writer, dpi=dpi)
            except Exception as e:
                # If saving fails, still show the animation interactively
                print('Warning: could not save animation:', e)
                plt.show()
        else:
            plt.show()
