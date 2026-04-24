import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xml.etree.ElementTree as ET
import os

adjacency_path  = "adjacency_coo.csv"
gexf_path       = "forceatlas2_rep_3.gexf"   # rename to match your file
steps_dir       = "gephi_colouring"     # directory with colours_1.csv, colours_2.csv, ...
output_file     = "polya_animation.mp4"

FPS             = 6
STEPS_PER_FRAME = 4   # increase to speed up, e.g. 3 = every 3rd step

# ── load graph ────────────────────────────────────────────────────────────────
edges = np.loadtxt(adjacency_path, delimiter=",", skiprows=1, dtype=int)
G = nx.Graph()
G.add_edges_from(edges)
num_nodes = G.number_of_nodes()

# ── load positions from gephi GEXF ───────────────────────────────────────────
def load_positions_from_gexf(gexf_path):
    tree = ET.parse(gexf_path)
    root = tree.getroot()
    ns = {
        "g":   "http://gexf.net/1.3",
        "viz": "http://gexf.net/1.3/viz"
    }
    pos = {}
    for node in root.findall(".//g:node", ns):
        node_id = int(node.get("id"))
        viz_pos = node.find("viz:position", ns)
        if viz_pos is not None:
            x = float(viz_pos.get("x"))
            y = float(viz_pos.get("y"))
            pos[node_id] = (x, y)
    return pos

print("Loading positions from GEXF...")
pos = load_positions_from_gexf(gexf_path)
print(f"Loaded positions for {len(pos)} nodes")

# ── load step colours ─────────────────────────────────────────────────────────
import re

step_files = sorted(
    (f for f in os.listdir(steps_dir) if f.endswith(".csv")),
    key=lambda f: int(re.search(r'\d+', f).group())  # sort by number, not alphabetically
)
step_files = step_files[::STEPS_PER_FRAME]

def load_colours(fname):
    colours = ["#282828"] * num_nodes
    with open(os.path.join(steps_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("node"):  # skip header if present
                continue
            node_id, colour = line.split(",")
            colours[int(node_id)] = "#FF0000" if colour == "red" else "#000000"
    return colours

# ── draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10), facecolor="#FFFFFF")
ax.set_facecolor("#FFFFFF")
ax.axis("off")

node_list = list(G.nodes())
edge_list = list(G.edges())

# draw edges once — they don't change
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#929292",
                       alpha=0.3, width=0.3, node_size=0)

# draw nodes — colours update each frame
init_colours = load_colours(step_files[0])
node_colours = [init_colours[n] for n in node_list]
nodes_drawn  = nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                       node_color=node_colours,
                                       node_size=1, ax=ax)

step_text = ax.text(0.01, 0.98, "", transform=ax.transAxes,
                    color="black", fontsize=11, va="top")

def update(frame_idx):
    colours      = load_colours(step_files[frame_idx])
    node_colours = [colours[n] for n in node_list]
    nodes_drawn.set_facecolor(node_colours)
    step_text.set_text(f"Step {frame_idx * STEPS_PER_FRAME}")
    return nodes_drawn, step_text

ani = animation.FuncAnimation(fig, update, frames=len(step_files),
                               interval=1000 // FPS, blit=True)

print(f"Saving to {output_file}...")
writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
ani.save(output_file, writer=writer, dpi=300)
print("Done.")