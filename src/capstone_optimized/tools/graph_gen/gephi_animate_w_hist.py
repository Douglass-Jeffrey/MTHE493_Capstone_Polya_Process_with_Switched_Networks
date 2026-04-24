import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xml.etree.ElementTree as ET
import os
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

adjacency_path  = "adjacency_coo.csv"
gexf_path       = "forceatlas2_rep_FINAL.gexf"   # rename to match your file
steps_dir       = "gephi_colouring"
output_file     = "high_frames_inj_polya_animation.mp4"

FPS             = 6*4
STEPS_PER_FRAME = 1   # increase to speed up, e.g. 3 = every 3rd step

# ── load graph ────────────────────────────────────────────────────────────────
print("Loading adjacency list...")
edges = np.loadtxt(adjacency_path, delimiter=",", skiprows=1, dtype=int)
G = nx.Graph()
G.add_edges_from(edges)
num_nodes = int(edges.max()) + 1

# ── load positions from gephi GEXF ───────────────────────────────────────────
def load_positions_from_gexf(gexf_path):
    tree = ET.parse(gexf_path)
    root = tree.getroot()
    # detect namespace from root tag e.g. {http://gexf.net/1.3}gexf
    ns_uri = root.tag.split("}")[0].strip("{")
    ns = {"g": ns_uri, "viz": ns_uri + "/viz"}
    pos = {}
    for node in root.findall(".//g:node", ns):
        node_id = int(node.get("id"))
        viz_pos = node.find("viz:position", ns)
        if viz_pos is not None:
            pos[node_id] = (float(viz_pos.get("x")), float(viz_pos.get("y")))
    return pos

print("Loading positions from GEXF...")
pos = load_positions_from_gexf(gexf_path)
print(f"Loaded positions for {len(pos)} nodes")

# ── load step files (sorted by number not alphabetically) ────────────────────
step_files = sorted(
    (f for f in os.listdir(steps_dir) if f.endswith(".csv")),
    key=lambda f: int(re.search(r'\d+', f).group())
)
step_files = step_files[::STEPS_PER_FRAME]
print(f"Animating {len(step_files)} frames...")

# update num_nodes in case CSV has more nodes than the edge list
with open(os.path.join(steps_dir, step_files[0])) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("node"):
            continue
        node_id, _ = line.split(",")
        num_nodes = max(num_nodes, int(node_id) + 1)
print(f"Total nodes: {num_nodes}")

# ── colour loader ─────────────────────────────────────────────────────────────
def load_colours(fname):
    colours = ["#000000"] * num_nodes
    with open(os.path.join(steps_dir, fname)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("node"):   # skip header if present
                continue
            node_id, colour = line.split(",")
            nid = int(node_id)
            if nid < num_nodes:
                colours[nid] = "#FF0000" if colour == "red" else "#000000"
    return colours

# ── load histogram data ───────────────────────────────────────────────────────
print("Loading histogram data...")
hist_data   = np.loadtxt("hist_data_gephi.csv", delimiter=",")
num_bins    = hist_data.shape[1]
bin_edges   = np.linspace(0, 1, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

row_sums = hist_data.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
hist_norm = hist_data / row_sums

# ── figure setup ──────────────────────────────────────────────────────────────
fig, ax_graph = plt.subplots(figsize=(14, 10), facecolor="white")
ax_graph.set_facecolor("white")
ax_graph.axis("off")

# ── inset histogram (bottom-right corner) ─────────────────────────────────────
ax_hist = inset_axes(ax_graph, width="28%", height="25%", loc="lower left",
                     borderpad=2)
ax_hist.set_facecolor("white")
ax_hist.patch.set_alpha(0.85)
ax_hist.set_xlim(0, 1)
ax_hist.set_ylim(0, hist_norm.max() * 1.1)
ax_hist.set_xlabel("P(red)", fontsize=7)
ax_hist.set_ylabel("Proportion", fontsize=7)
ax_hist.tick_params(labelsize=6)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

bars = ax_hist.bar(bin_centers, hist_norm[0],
                   width=bin_edges[1] - bin_edges[0],
                   color="#0000FF", alpha=0.8, edgecolor="none")

# ── draw edges once (they don't change) ──────────────────────────────────────
print("Drawing edges...")
nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="#AAAAAA",
                       alpha=0.3, width=0.05/2, node_size=0)

# ── initial node colours ──────────────────────────────────────────────────────
init_colours = load_colours(step_files[0])
node_list    = list(G.nodes())
node_colours = [init_colours[n] for n in node_list]
nodes_drawn  = nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                       node_color=node_colours,
                                       node_size=1/2, ax=ax_graph)

step_text = ax_graph.text(0.01, 0.98, "", transform=ax_graph.transAxes,
                           color="black", fontsize=11, va="top")

plt.tight_layout()

# ── animation update function ─────────────────────────────────────────────────
def update(frame_idx):
    # update graph colours
    colours      = load_colours(step_files[frame_idx])
    node_colours = [colours[n] for n in node_list]
    nodes_drawn.set_facecolor(node_colours)
    step_text.set_text(f"Step {frame_idx * STEPS_PER_FRAME}")

    # update histogram
    actual_step = frame_idx * STEPS_PER_FRAME
    if actual_step < len(hist_norm):
        for bar, h in zip(bars, hist_norm[actual_step]):
            bar.set_height(h)

    return [nodes_drawn, step_text] + list(bars)

ani = animation.FuncAnimation(fig, update, frames=len(step_files),
                               interval=1000 // FPS, blit=True)

print(f"Saving to {output_file}...")
writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
ani.save(output_file, writer=writer, dpi=300)
print("Done.")