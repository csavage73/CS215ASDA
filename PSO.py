import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Fixed node positions (x, y) in a logical layout
nodes = {
    'A': (0,0), 'B': (3,0), 'C': (0,2),
    'D': (1,5), 'E': (3,3), 'F': (5,3),
    'G': (3,6), 'H': (5,7), 'I': (6,9),
    'J': (6,5), 'K': (9,3), 'L': (10,1),
    'M': (11,5), 'N': (12,2), 'O': (6,0), 
    'P': (6,2), 'Q': (8,7), 'R': (10,10),
    'S': (15,10), 'T': (14,5), 'U': (16,8),
    'V': (3,11), 'W': (4,13), 'X': (5,12),
    'Y': (10,12), 'Z': (7,13)
}

# Fixed edges with explicit weights (distances)
edges = [
    ('A','B',0), ('A','C',0), ('B','C',0), ('B','E',0), ('B','F',0),
    ('B','O',0), ('C','D',0), ('D','E',0), ('D','G',0), ('D','V',0), 
    ('E','F',0), ('E','G',0), ('E','H',0), ('E','J',0), ('F','J',0), 
    ('F','P',0), ('G','H',0), ('G','H',0), ('G','V',0), ('H','I',0), 
    ('H','J',0), ('I','J',0), ('I','Q',0), ('I','R',0), ('I','V',0), 
    ('I','X',0), ('I','Y',0), ('J','K',0), ('J','P',0), ('J','Q',0), 
    ('K','L',0), ('K','M',0), ('K','P',0), ('K','Q',0), ('K','R',0), 
    ('L','N',0), ('L','O',0), ('M','N',0), ('M','R',0), ('M','S',0), 
    ('M','T',0), ('N','T',0), ('O','P',0), ('Q','R',0), ('R','S',0), 
    ('R','Y',0), ('S','T',0), ('S','U',0), ('S','Y',0), ('T','U',0), 
    ('V','W',0), ('V','X',0), ('W','X',0), ('W','Z',0), ('X','Y',0), 
    ('X','Z',0), ('Y','Z',0),
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

def decode_path(priority, source, target):
    """Greedy path construction driven by priority vector."""
    path    = [source]
    visited = {source}
    current = source
    while current != target:
        neighbors = [nb for nb in adj[current] if nb not in visited]
        if not neighbors:
            return None                 # dead end
        current = max(neighbors, key=lambda n: priority[node_idx[n]])
        path.append(current)
        visited.add(current)
        if len(path) > n_nodes:         # cycle guard
            return None
    return path

def path_cost(path):
    return sum(adj[path[i]][path[i+1]] for i in range(len(path) - 1))

def fitness(priority, source, target):
    path = decode_path(priority, source, target)
    return path_cost(path) if path else float('inf')

# ── Hyper-parameters ──
NUM_PARTICLES = 40
MAX_ITER      = 200
W             = 0.6   # inertia weight
C1            = 1.5   # cognitive (personal best) coefficient
C2            = 1.5   # social (global best) coefficient
V_MAX         = 0.3   # velocity clamp

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

edge_labels = {(u, v): f"{adj[u][v]:.1f}" for u, v in path_edges}
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
