import random
import math
import time
import networkx as nx
import matplotlib.pyplot as plt


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_spread_out_positions(num_nodes, width, height, min_dist, seed=None):
    random.seed(seed)
    positions = {}
    attempts_limit = 10000

    positions["N01"] = (width * 0.1, height * 0.1)
    positions[f"N{num_nodes:02}"] = (width * 0.9, height * 0.9)

    for i in range(num_nodes):
        node_name = f"N{i+1:02}"

        if node_name in positions:
            continue

        placed = False
        attempts = 0

        while not placed and attempts < attempts_limit:
            x = random.randint(0, width)
            y = random.randint(0, height)

            too_close = False
            for other_pos in positions.values():
                if euclidean_distance((x, y), other_pos) < min_dist:
                    too_close = True
                    break

            if not too_close:
                positions[node_name] = (x, y)
                placed = True

            attempts += 1

        if not placed:
            raise ValueError("Could not place all nodes. Try lowering num_nodes or min_dist.")

    return positions


def edge_weight(congestion_chance):
    # Congested = high weight (avoided), not congested = low weight (preferred)
    if random.randint(0, 99) < congestion_chance:
        return random.randint(10, 20)   # congested
    return random.randint(1, 5)         # not congested


def generate_random_map(num_nodes=500, width=600, height=400, extra_edges=100, min_dist=5, seed=42, congestion_chance=30):
    positions = generate_spread_out_positions(num_nodes, width, height, min_dist, seed)
    G = nx.Graph()

    for node in positions:
        G.add_node(node)

    node_list = list(positions.keys())
    connected = {node_list[0]}
    unconnected = set(node_list[1:])

    # Step 1: Build spanning tree to guarantee full connectivity
    while unconnected:
        best_pair = None
        best_dist = float("inf")

        for u in connected:
            for v in unconnected:
                dist = euclidean_distance(positions[u], positions[v])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (u, v)

        u, v = best_pair
        G.add_edge(u, v, weight=edge_weight(congestion_chance))
        connected.add(v)
        unconnected.remove(v)

    # Step 2: Add extra nearby edges for more path options
    possible_edges = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            u = node_list[i]
            v = node_list[j]
            if not G.has_edge(u, v):
                dist = euclidean_distance(positions[u], positions[v])
                possible_edges.append((dist, u, v))

    possible_edges.sort(key=lambda x: x[0])

    added = 0
    for dist, u, v in possible_edges:
        if added >= extra_edges:
            break
        G.add_edge(u, v, weight=edge_weight(congestion_chance))
        added += 1

    # Step 3: Ensure every node has at least degree 2
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes):
            while G.degree[node] < 2:
                candidates = []
                for other in node_list:
                    if other != node and not G.has_edge(node, other):
                        dist = euclidean_distance(positions[node], positions[other])
                        candidates.append((dist, other))

                candidates.sort(key=lambda x: x[0])

                if not candidates:
                    break

                _, nearest = candidates[0]
                G.add_edge(node, nearest, weight=edge_weight(congestion_chance))
                changed = True

    return G, positions


def nx_to_adj_dict(G):
    graph_dict = {}

    for node in G.nodes:
        graph_dict[node] = {}

    for u, v, data in G.edges(data=True):
        w = data["weight"]
        graph_dict[u][v] = w
        graph_dict[v][u] = w

    return graph_dict


def bellman_ford(graph, source):
    distances = {v: float("inf") for v in graph}
    predecessors = {v: None for v in graph}
    distances[source] = 0

    for _ in range(len(graph) - 1):
        updated = False
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    updated = True
        if not updated:
            break

    return distances, predecessors


def reconstruct_path(predecessors, source, target):
    path = []
    current = target

    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()

    if not path or path[0] != source:
        return None

    return path


def draw_graph_with_path(G, positions, path, source, target, cost):
    plt.figure(figsize=(16, 10))

    nx.draw_networkx_edges(G, positions, alpha=0.6, width=1.5)

    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, positions, edgelist=path_edges, width=3.5, edge_color="red")

    large_graph = len(G.nodes) > 150

    node_size = 80 if large_graph else 350
    font_size = 5 if large_graph else 8
    nx.draw_networkx_nodes(G, positions, node_size=node_size)
    nx.draw_networkx_labels(G, positions, font_color='white', font_size=font_size)

    nx.draw_networkx_nodes(G, positions, nodelist=[source], node_color="green", node_size=node_size + 50)
    nx.draw_networkx_nodes(G, positions, nodelist=[target], node_color="orange", node_size=node_size + 50)

    if not large_graph:
        for u, v, data in G.edges(data=True):
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            plt.text(
                (x1 + x2) / 2, (y1 + y2) / 2,
                str(data["weight"]),
                fontsize=7, ha="center", va="center", color="black", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.15)
            )

    plt.title(f"Bellman-Ford Shortest Path: {source} -> {target} | Total Cost = {cost}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(-5, max(x for x, _ in positions.values()) + 5)
    plt.ylim(-5, max(y for _, y in positions.values()) + 5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CONGESTION_CHANCE = 23  # % chance an edge is congested (high weight). Change this value.
    NUM_NODES = 200         # Number of nodes in the graph

    # --- Static graph (same layout every run) ---
    SEED = 42
    # --- Dynamic graph (different layout every run) --- uncomment the line below and comment out the line above
    #SEED = random.randint(0, 999999)

    G, positions = generate_random_map(
        num_nodes=NUM_NODES,
        width=600,
        height=400,
        extra_edges=100,
        min_dist=5,
        seed=SEED      
    )

    print(f"Total nodes: {G.number_of_nodes()}")

    graph_dict = nx_to_adj_dict(G)

    SOURCE = "N01"
    TARGET = f"N{NUM_NODES:02}"

    start_time = time.perf_counter()
    distances, predecessors = bellman_ford(graph_dict, SOURCE)
    shortest_path = reconstruct_path(predecessors, SOURCE, TARGET)
    end_time = time.perf_counter()

    print(f"Source: {SOURCE}")
    print(f"Target: {TARGET}")
    print(f"Bellman-Ford shortest path:")
    print(" -> ".join(shortest_path))
    print(f"Total cost: {distances[TARGET]}")
    print(f"Algorithm runtime: {end_time - start_time:.6f} seconds")

    draw_graph_with_path(G, positions, shortest_path, SOURCE, TARGET, distances[TARGET])
