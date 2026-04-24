import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

adjacency_path = "adjacency_coo.csv"
steps_dir      = "gephi_colouring"
output_gexf    = "network_dynamic.gexf"

# ── load adjacency list ───────────────────────────────────────────────────────
edges = np.loadtxt(adjacency_path, delimiter=",", skiprows=1, dtype=int)
num_nodes = int(edges.max()) + 1

# ── discover available steps ──────────────────────────────────────────────────
step_files = sorted(f for f in os.listdir(steps_dir) if f.endswith(".csv"))
num_steps  = len(step_files)
print(f"Found {num_steps} step files, {num_nodes} nodes, {len(edges)} edges")

# ── build colour timeline per node ────────────────────────────────────────────
# colour_at[node][step] = "red" | "black"
colour_at = [[] for _ in range(num_nodes)]
for step_i, fname in enumerate(step_files):
    path = os.path.join(steps_dir, fname)
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            node_id, colour = line.strip().split(",")
            colour_at[int(node_id)].append(colour)

# collapse into spans: (start, end, colour)
# Gephi timeline uses integer step numbers as timestamps
def to_spans(colours):
    spans = []
    if not colours:
        return spans
    cur_colour = colours[0]
    cur_start  = 0
    for t, c in enumerate(colours[1:], start=1):
        if c != cur_colour:
            spans.append((cur_start, t, cur_colour))
            cur_colour = c
            cur_start  = t
    spans.append((cur_start, len(colours), cur_colour))
    return spans

# ── build GEXF XML ────────────────────────────────────────────────────────────
GEXF_NS  = "http://gexf.net/1.3"
VIZ_NS   = "http://gexf.net/1.3/viz"

gexf = ET.Element("gexf", {
    "xmlns":     GEXF_NS,
    "xmlns:viz": VIZ_NS,
    "version":   "1.3"
})

graph_el = ET.SubElement(gexf, "graph", {
    "defaultedgetype": "undirected",
    "mode":            "dynamic",
    "timeformat":      "integer",
    "start":           "0",
    "end":             str(num_steps)
})

# declare node attribute: colour (partition)
attrs_el = ET.SubElement(graph_el, "attributes", {"class": "node", "mode": "dynamic"})
ET.SubElement(attrs_el, "attribute", {"id": "0", "title": "colour", "type": "string"})

# ── nodes ─────────────────────────────────────────────────────────────────────
RED_HEX   = {"r": "220", "g": "50",  "b": "47"}
BLACK_HEX = {"r": "40",  "g": "40",  "b": "40"}

nodes_el = ET.SubElement(graph_el, "nodes")
for node_id in range(num_nodes):
    node_el = ET.SubElement(nodes_el, "node", {"id": str(node_id), "label": str(node_id)})
    spans = to_spans(colour_at[node_id])

    # dynamic attribute values (text label used by partitions/filters)
    attvals_el = ET.SubElement(node_el, "attvalues")
    for start, end, colour in spans:
        ET.SubElement(attvals_el, "attvalue", {
            "for":   "0",
            "value": colour,
            "start": str(start),
            "end":   str(end)
        })

    # dynamic viz:color (drives the actual rendered colour in Gephi)
    for start, end, colour in spans:
        hex_vals = RED_HEX if colour == "red" else BLACK_HEX
        color_el = ET.SubElement(node_el, f"{{{VIZ_NS}}}color", {
            **hex_vals,
            "start": str(start),
            "end":   str(end)
        })

# ── edges ─────────────────────────────────────────────────────────────────────
edges_el = ET.SubElement(graph_el, "edges")
for i, (src, dst) in enumerate(edges):
    ET.SubElement(edges_el, "edge", {
        "id":     str(i),
        "source": str(src),
        "target": str(dst)
    })

# ── write pretty-printed GEXF ─────────────────────────────────────────────────
raw    = ET.tostring(gexf, encoding="unicode")
pretty = minidom.parseString(raw).toprettyxml(indent="  ")
with open(output_gexf, "w", encoding="utf-8") as f:
    f.write(pretty)

print(f"Written: {output_gexf}")