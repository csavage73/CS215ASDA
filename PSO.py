import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Fixed node positions (x, y) in a logical layout
nodes = {
    'A': (0,0),  'B': (3,0),  'C': (0,2),
    'D': (1,5),  'E': (3,3),  'F': (5,3),
    'G': (3,6),  'H': (5,7),  'I': (6,9),
    'J': (6,5),  'K': (9,3),  'L': (10,1),
    'M': (11,5), 'N': (12,2), 'O': (6,0), 
    'P': (6,2),  'Q': (8,7),  'R': (10,10),
    'S': (15,10),'T': (14,5), 'U': (16,8),
    'V': (3,11), 'W': (4,13), 'X': (5,12),
    'Y': (10,12),'Z': (7,13)
}

# Fixed edges with explicit weights (distances)
edges = [
    ('A','B',24), ('A','C',65), ('B','C',34), ('B','E',999), ('B','F',65),
    ('B','O',13), ('C','D',22), ('D','E',44), ('D','G',54), ('D','V',34),
    ('E','F',65), ('E','G',99), ('E','H',78), ('E','J',39), ('F','J',54),
    ('F','P',44), ('G','H',28), ('G','V',73), ('H','I',32), 
    ('H','J',22), ('I','J',47), ('I','Q',23), ('I','R',54), ('I','V',32),
    ('I','X',53), ('I','Y',65), ('J','K',2), ('J','P',42), ('J','Q',43),
    ('K','L',76), ('K','M',98), ('K','P',45), ('K','Q',90), ('K','R',65),
    ('L','N',1), ('L','O',46), ('M','N',35), ('M','R',14), ('M','S',78),
    ('M','T',23), ('N','T',67), ('O','P',52), ('Q','R',45), ('R','S',65),
    ('R','Y',88), ('S','T',98), ('S','U',89), ('S','Y',34), ('T','U',95),
    ('V','W',21), ('V','X',34), ('W','X',56), ('W','Z',87), ('X','Y',32),
    ('X','Z',34), ('Y','Z',43)
]

G = nx.Graph()
G.add_nodes_from(nodes.keys())
for u, v, w in edges:
    G.add_edge(u, v, weight=w)




SOURCE = 'A'
TARGET = 'Z'

node_list = list(nodes.keys())          # fixed ordering for indexing
n_nodes   = len(node_list)
node_idx  = {n: i for i, n in enumerate(node_list)}

adj = nx.to_dict_of_dicts(G)

MAX_EDGE_WEIGHT = 9.0   # used to normalise edge cost into [0, 1]

def decode_path(priority, source, target):
    """Greedy path construction driven by priority vector and edge cost.

    Selection score = priority[neighbour] - normalised_edge_weight
    This lets the PSO learn high-priority assignments for cheap-path nodes
    while the decoder itself also prefers lower-cost edges at each step.
    """
    path    = [source]
    visited = {source}
    current = source
    while current != target:
        neighbors = [nb for nb in adj[current] if nb not in visited]
        if not neighbors:
            return None                 # dead end
        current = max(
            neighbors,
            key=lambda n: priority[node_idx[n]]
                          - adj[current][n]['weight'] / MAX_EDGE_WEIGHT
        )
        path.append(current)
        visited.add(current)
        if len(path) > n_nodes:         # cycle guard
            return None
    return path

def path_cost(path):
    return sum(adj[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

def fitness(priority, source, target):
    path = decode_path(priority, source, target)
    return path_cost(path) if path else float('inf')

# ── Hyper-parameters ──
NUM_PARTICLES = 100
MAX_ITER      = 1000
W_MAX         = 0.9   # inertia weight – starts high (exploration)
W_MIN         = 0.4   #                – decays to this (exploitation)
C1            = 1.5   # cognitive (personal best) coefficient
C2            = 1.5   # social (global best) coefficient
V_MAX         = 0.5   # velocity clamp

np.random.seed(42)
positions  = np.random.uniform(0, 1, (NUM_PARTICLES, n_nodes))
velocities = np.random.uniform(-V_MAX, V_MAX, (NUM_PARTICLES, n_nodes))

pbest        = positions.copy()
pbest_scores = np.array([fitness(p, SOURCE, TARGET) for p in positions])

gbest_idx   = int(np.argmin(pbest_scores))
gbest       = pbest[gbest_idx].copy()
gbest_score = pbest_scores[gbest_idx]

history = []   # track best cost per iteration

for iteration in range(MAX_ITER):
    W = W_MAX - (W_MAX - W_MIN) * iteration / MAX_ITER   # linearly decay inertia
    for i in range(NUM_PARTICLES):
        r1 = np.random.rand(n_nodes)
        r2 = np.random.rand(n_nodes)

        velocities[i] = (
            W  * velocities[i]
            + C1 * r1 * (pbest[i] - positions[i])
            + C2 * r2 * (gbest   - positions[i])
        )
        velocities[i] = np.clip(velocities[i], -V_MAX, V_MAX)
        positions[i]  = np.clip(positions[i] + velocities[i], 0, 1)

        score = fitness(positions[i], SOURCE, TARGET)
        if score < pbest_scores[i]:
            pbest[i]        = positions[i].copy()
            pbest_scores[i] = score
            if score < gbest_score:
                gbest       = positions[i].copy()
                gbest_score = score

    history.append(gbest_score)

best_path = decode_path(gbest, SOURCE, TARGET)
print(f"Source: {SOURCE}  →  Target: {TARGET}")
print(f"Best path  : {' → '.join(best_path)}")
print(f"Total cost : {gbest_score:.4f}")

# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────
edge_list = list(G.edges())
fig, (ax_graph, ax_conv) = plt.subplots(1, 2, figsize=(18, 7))

# ── Left: Graph with best path highlighted ──
path_edges = list(zip(best_path[:-1], best_path[1:]))
other_edges = [(u, v) for u, v in edge_list if
               (u, v) not in path_edges and (v, u) not in path_edges]

nx.draw_networkx_nodes(G, nodes, node_size=500, node_color='#534AB7', ax=ax_graph)
nx.draw_networkx_labels(G, nodes, font_color='white', font_size=11,
                        font_weight='bold', ax=ax_graph)
nx.draw_networkx_edges(G, nodes, edgelist=other_edges,
                       width=1.2, alpha=0.4, edge_color='#888780', ax=ax_graph)
nx.draw_networkx_edges(G, nodes, edgelist=path_edges,
                       width=3.5, edge_color='#E84545', ax=ax_graph,
                       label='Best path')

# Highlight source and target nodes
nx.draw_networkx_nodes(G, nodes, nodelist=[SOURCE, TARGET],
                       node_size=650, node_color='#E84545', ax=ax_graph)

edge_labels = {(u, v): f"{adj[u][v]['weight']:.1f}" for u, v in path_edges}
nx.draw_networkx_edge_labels(G, nodes, edge_labels=edge_labels,
                             font_size=7, ax=ax_graph)

ax_graph.set_title(
    f"PSO Shortest Path  {SOURCE} → {TARGET}\n"
    f"Path: {' → '.join(best_path)}   Cost: {gbest_score:.2f}",
    fontsize=11
)
ax_graph.set_xlim(-1, 17)
ax_graph.set_ylim(-1, 15)
ax_graph.grid(True, linestyle='--', alpha=0.4)
ax_graph.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# ── Right: Convergence curve ──
ax_conv.plot(history, color='#534AB7', linewidth=2)
ax_conv.set_title('PSO Convergence', fontsize=11)
ax_conv.set_xlabel('Iteration')
ax_conv.set_ylabel('Best Path Cost')
ax_conv.grid(True, linestyle='--', alpha=0.4)


















# Visualize
fig, ax = plt.subplots(figsize=(10, 7))

nx.draw_networkx_nodes(G, nodes, node_size=500, node_color='#534AB7', ax=ax)
nx.draw_networkx_labels(G, nodes, font_color='white', font_size=11, font_weight='bold', ax=ax)
nx.draw_networkx_edges(G, nodes, width=1.2, alpha=0.5, edge_color='#888780', ax=ax)
nx.draw_networkx_edge_labels(
    G, nodes,
    edge_labels={(u, v): w for u, v, w in edges},
    font_size=8, ax=ax
)


ax.set_xlim(-1, 17)
ax.set_ylim(-1, 15)
ax.grid(True, linestyle='--', alpha=0.4)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#plt.tight_layout()
plt.tight_layout()
plt.show()
