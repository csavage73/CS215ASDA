import random
import math
import networkx as nx
import matplotlib.pyplot as plt


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_spread_out_positions(num_nodes, width, height, min_dist, seed=None):
    random.seed(seed)
    positions = {}
    attempts_limit = 10000

    # Force N01 and N100 far apart
    positions["N01"] = (width * 0.1, height * 0.1)
    positions[f"N{num_nodes:02}"] = (width * 0.9, height * 0.9)

    for i in range(num_nodes):
        node_name = f"N{i+1:02}"

        # Skip nodes already manually placed
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


def generate_random_map(num_nodes=100, width=200, height=130, extra_edges=15, min_dist=7, seed=42):
    positions = generate_spread_out_positions(num_nodes, width, height, min_dist, seed)
    G = nx.Graph()

    for node in positions:
        G.add_node(node)

    node_list = list(positions.keys())
    connected = {node_list[0]}
    unconnected = set(node_list[1:])

    # Step 1: Force full connectivity
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
        G.add_edge(u, v, weight=random.randint(1, 20))
        connected.add(v)
        unconnected.remove(v)

    # Step 2: Add extra nearby edges
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
        G.add_edge(u, v, weight=random.randint(1, 20))
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

                dist, nearest = candidates[0]
                G.add_edge(node, nearest, weight=random.randint(1, 20))
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

    for u in graph:
        for v, weight in graph[u].items():
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                raise ValueError("Negative-weight cycle detected")

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

    # Draw normal edges first
    nx.draw_networkx_edges(G, positions, alpha=0.6, width=1.5)

    # Highlight Bellman-Ford path
    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G,
            positions,
            edgelist=path_edges,
            width=3.5,
            edge_color="red"
        )

    # Draw nodes
    nx.draw_networkx_nodes(G, positions, node_size=350)
    nx.draw_networkx_labels(G, positions, font_color='white', font_size=8)

    # Highlight source and target
    nx.draw_networkx_nodes(G, positions, nodelist=[source], node_color="green", node_size=400)
    nx.draw_networkx_nodes(G, positions, nodelist=[target], node_color="orange", node_size=400)

    # Draw weights centered on the lines
    for u, v, data in G.edges(data=True):
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        plt.text(
            mx,
            my,
            str(data["weight"]),
            fontsize=7,
            ha="center",
            va="center",
            color="black",
            zorder=10,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=0.15
            )
        )

    plt.title(f"Bellman-Ford Shortest Path From {source} - {target} | Total Cost = {cost}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(-5, max(x for x, y in positions.values()) + 5)
    plt.ylim(-5, max(y for x, y in positions.values()) + 5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    G, positions = generate_random_map(
        num_nodes=100,
        width=170,
        height=110,
        extra_edges=15,
        min_dist=9,
        seed=42
    )

    graph_dict = nx_to_adj_dict(G)

    SOURCE = "N01"
    TARGET = "N100"

    distances, predecessors = bellman_ford(graph_dict, SOURCE)
    shortest_path = reconstruct_path(predecessors, SOURCE, TARGET)

    print(f"Source: {SOURCE}")
    print(f"Target: {TARGET}")
    print("Bellman-Ford shortest path:")
    print(" -> ".join(shortest_path))
    print(f"Total cost: {distances[TARGET]}")

    draw_graph_with_path(G, positions, shortest_path, SOURCE, TARGET, distances[TARGET])