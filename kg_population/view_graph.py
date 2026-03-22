import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gexf("memory/knowledge_graph.gexf")

# Colour nodes by type
color_map = {
    "Paper":               "#4A90D9",
    "Method":              "#E67E22",
    "Dataset":             "#27AE60",
    "Metric":              "#8E44AD",
    "Task":                "#E74C3C",
    "LimitationStatement": "#95A5A6",
    "FutureWork":          "#F1C40F",
}
colors = [color_map.get(d.get("node_type", ""), "#CCCCCC") 
          for _, d in G.nodes(data=True)]

plt.figure(figsize=(18, 12))
pos = nx.spring_layout(G, k=2, seed=42)
nx.draw_networkx(G, pos, node_color=colors, node_size=300,
                 font_size=6, arrows=True, with_labels=True)
plt.axis("off")
plt.tight_layout()
plt.savefig("memory/graph_preview.png", dpi=150)
plt.show()
print("Saved to memory/graph_preview.png")