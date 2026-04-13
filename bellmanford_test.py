import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ── Bellman-Ford ────────────────────────────────────────────────────────────
def bellman_ford(graph, source):
    distances = {v: float('inf') for v in graph}
    predecessors = {v: None for v in graph}
    distances[source] = 0

    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

    for u in graph:
        for v, weight in graph[u].items():
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                raise ValueError("Negative-weight cycle detected")

    return distances, predecessors


def get_shortest_path_edges(predecessors):
    return {(pred, node) for node, pred in predecessors.items() if pred is not None}


# ── Style constants ──────────────────────────────────────────────────────────
STYLE = {
    'bg':           '#0F1117',   # near-black background
    'panel':        '#1A1D27',   # slightly lighter panel
    'grid':         '#2A2D3A',   # subtle grid lines
    'text':         '#E8E8F0',   # primary text
    'subtext':      '#9090A8',   # secondary text
    'node_default': '#3A3D50',   # unvisited node
    'edge_default': '#2E3145',   # background edge color
}

CORRIDOR_COLORS = {
    'I-40':  {'main': '#FF6B6B', 'glow': '#FF2D2D'},  # red
    'I-440': {'main': '#4ECDC4', 'glow': '#00FFF0'},  # teal
    'I-540': {'main': '#FFE66D', 'glow': '#FFD700'},  # gold
    'Other': {'main': '#A78BFA', 'glow': '#7C3AED'},  # purple
}


# ── Pull OSMnx road network ──────────────────────────────────────────────────
print("Fetching Raleigh-Cary road network...")
G_osm = ox.graph_from_place("Raleigh, North Carolina, USA", network_type='drive')

HIGHWAY_TYPES = {'motorway', 'motorway_link', 'trunk', 'trunk_link'}

filtered_edges = [
    (u, v, data) for u, v, data in G_osm.edges(data=True)
    if data.get('highway') in HIGHWAY_TYPES
    or (isinstance(data.get('highway'), list)
        and any(h in HIGHWAY_TYPES for h in data['highway']))
]

filtered_nodes = set()
for u, v, _ in filtered_edges:
    filtered_nodes.add(u)
    filtered_nodes.add(v)

graph_dict = {n: {} for n in filtered_nodes}
for u, v, data in filtered_edges:
    weight = data.get('length', 1)
    if v not in graph_dict[u] or weight < graph_dict[u][v]:
        graph_dict[u][v] = weight


# ── Corridor tagging ─────────────────────────────────────────────────────────
CORRIDORS = {
    'I-40':  {'I-40', 'Interstate 40'},
    'I-440': {'I-440', 'Interstate 440', 'Capital Beltline'},
    'I-540': {'I-540', 'Interstate 540', 'Triangle Expressway'},
}

def get_corridor(data):
    ref  = data.get('ref', '')
    name = data.get('name', '')
    if isinstance(ref, list):
        ref = ' '.join(ref)
    for corridor, tags in CORRIDORS.items():
        if any(t in ref or t in name for t in tags):
            return corridor
    return 'Other'

corridor_graphs = {c: {} for c in list(CORRIDORS.keys()) + ['Other']}
for u, v, data in filtered_edges:
    corridor = get_corridor(data)
    weight   = data.get('length', 1)
    if u not in corridor_graphs[corridor]:
        corridor_graphs[corridor][u] = {}
    if v not in corridor_graphs[corridor][u] or weight < corridor_graphs[corridor][u][v]:
        corridor_graphs[corridor][u][v] = weight


# ── Named source nodes via nearest-node lookup ───────────────────────────────
INTERCHANGE_COORDS = {
    'I-40':  (35.8878, -78.7876),
    'I-440': (35.8321, -78.6499),
    'I-540': (35.8897, -78.7868),
}

named_sources = {}
for corridor, (lat, lon) in INTERCHANGE_COORDS.items():
    named_sources[corridor] = ox.distance.nearest_nodes(G_osm, X=lon, Y=lat)


# ── Run Bellman-Ford ─────────────────────────────────────────────────────────
corridor_results = {}
for corridor, subgraph in corridor_graphs.items():
    if len(subgraph) == 0:
        continue
    source = named_sources.get(corridor, next(iter(subgraph)))
    try:
        distances, predecessors = bellman_ford(subgraph, source)
        corridor_results[corridor] = {
            'distances':    distances,
            'predecessors': predecessors,
            'source':       source,
            'subgraph':     subgraph,
        }
        print(f"{corridor}: {len(subgraph)} nodes processed")
    except ValueError as e:
        print(f"{corridor}: {e}")


# ── Node positions ───────────────────────────────────────────────────────────
node_positions = {
    n: (G_osm.nodes[n]['x'], G_osm.nodes[n]['y'])
    for n in filtered_nodes if n in G_osm.nodes
}


# ── Draw each corridor ───────────────────────────────────────────────────────
corridors_to_draw = [c for c in ['I-40', 'I-440', 'I-540', 'Other']
                     if c in corridor_results]

fig, axes = plt.subplots(1, len(corridors_to_draw),
                         figsize=(7 * len(corridors_to_draw), 9))
fig.patch.set_facecolor(STYLE['bg'])

if len(corridors_to_draw) == 1:
    axes = [axes]

for ax, corridor in zip(axes, corridors_to_draw):
    result   = corridor_results[corridor]
    subgraph = result['subgraph']
    preds    = result['predecessors']
    source   = result['source']
    colors   = CORRIDOR_COLORS.get(corridor, CORRIDOR_COLORS['Other'])

    g = nx.DiGraph()
    for u in subgraph:
        for v, w in subgraph[u].items():
            g.add_edge(u, v, weight=w)

    pos             = {n: node_positions[n] for n in subgraph if n in node_positions}
    shortest_edges  = get_shortest_path_edges(preds)
    normal_edges    = [(u, v) for u, v in g.edges() if (u, v) not in shortest_edges]
    highlight_edges = [(u, v) for u, v in g.edges() if (u, v) in shortest_edges]

    ax.set_facecolor(STYLE['panel'])

    # Subtle grid
    ax.grid(True, linestyle='--', linewidth=0.4, color=STYLE['grid'], alpha=0.6, zorder=0)

    # Background (non-shortest) edges
    nx.draw_networkx_edges(
        g, pos,
        edgelist=normal_edges,
        width=0.6, alpha=0.25,
        edge_color=STYLE['edge_default'],
        arrows=False, ax=ax
    )

    # Glow layer for shortest path edges (wide + transparent = soft glow)
    nx.draw_networkx_edges(
        g, pos,
        edgelist=highlight_edges,
        width=8, alpha=0.15,
        edge_color=colors['glow'],
        arrows=False, ax=ax
    )

    # Core shortest path edges
    nx.draw_networkx_edges(
        g, pos,
        edgelist=highlight_edges,
        width=2.5, alpha=0.95,
        edge_color=colors['main'],
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        ax=ax
    )

    # All nodes (small, muted)
    nx.draw_networkx_nodes(
        g, pos,
        node_size=12,
        node_color=STYLE['node_default'],
        alpha=0.7, ax=ax
    )

    # Nodes ON the shortest path tree (larger, colored)
    sp_nodes = {n for edge in shortest_edges for n in edge} | {source}
    sp_nodes_in_pos = [n for n in sp_nodes if n in pos]
    nx.draw_networkx_nodes(
        g, pos,
        nodelist=sp_nodes_in_pos,
        node_size=40,
        node_color=colors['main'],
        alpha=1.0, ax=ax
    )

    # Source node (distinct — white with colored border)
    if source in pos:
        ax.scatter(
            *pos[source],
            s=220, zorder=10,
            facecolors='white',
            edgecolors=colors['main'],
            linewidths=2.5
        )
        ax.annotate(
            f"Source\n({corridor})",
            xy=pos[source],
            xytext=(12, 12), textcoords='offset points',
            fontsize=7.5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor=colors['main'], alpha=0.75, edgecolor='none'),
            path_effects=[pe.withStroke(linewidth=1.5, foreground=STYLE['bg'])]
        )

    # Title per panel
    ax.set_title(
        corridor,
        fontsize=15, fontweight='bold',
        color=colors['main'], pad=12
    )
    ax.set_xlabel('Longitude', fontsize=8, color=STYLE['subtext'])
    ax.set_ylabel('Latitude',  fontsize=8, color=STYLE['subtext'])
    ax.tick_params(colors=STYLE['subtext'], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE['grid'])

    # Per-panel legend
    legend_elements = [
        Line2D([0], [0], color=colors['main'],     linewidth=2.5, label='Shortest path tree'),
        Line2D([0], [0], color=STYLE['edge_default'], linewidth=0.8, alpha=0.5, label='Other edges'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=colors['main'], markersize=8, label='Source node'),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5,
              facecolor=STYLE['bg'], edgecolor=STYLE['grid'],
              labelcolor=STYLE['text'], loc='lower right')


# ── Global title & export ────────────────────────────────────────────────────
fig.suptitle(
    "Bellman-Ford Shortest Path Trees\nRaleigh–Cary Interstate Corridors",
    fontsize=17, fontweight='bold',
    color=STYLE['text'], y=1.01
)

plt.tight_layout(pad=2.0)
plt.savefig(
    "raleigh_bellman_ford.png",
    dpi=300,
    bbox_inches='tight',
    facecolor=STYLE['bg']
)
print("Saved: raleigh_bellman_ford.png")
plt.show()