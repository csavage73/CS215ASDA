import random
import math
import networkx as nx
import matplotlib.pyplot as plt


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_spread_out_positions(num_nodes, width, height, min_dist, seed=None):
    random.seed(seed)
    positions = {}
    # Force N01 and N100 far apart
    positions["N01"] = (width * 0.1, height * 0.1)
    positions[f"N{num_nodes:02}"] = (width * 0.9, height * 0.9)
    attempts_limit = 10000

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

    # Step 3: Remove dead-end branches
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


def draw_graph(G, positions):
    plt.figure(figsize=(16, 10))

    nx.draw_networkx_nodes(G, positions, node_size=350)
    nx.draw_networkx_labels(G, positions, font_color='white', font_size=8)
    nx.draw_networkx_edges(G, positions, alpha=0.6, width=1.5)

    # Draw weights manually
    for u, v, data in G.edges(data=True):
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)

        # Offset label a little off the edge
        if length != 0:
            offset_x = -dy / length * 1.2
            offset_y = dx / length * 1.2
        else:
            offset_x = 0
            offset_y = 0

        plt.text(
            mx + offset_x,
            my + offset_y,
            str(data["weight"]),
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8)
        )

    plt.title("Random Map Layout From N01 - N100")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(-5, max(x for x, y in positions.values()) + 5)
    plt.ylim(-5, max(y for x, y in positions.values()) + 5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Generating random map...")
    G, positions = generate_random_map(
        num_nodes=100,
        width=170,
        height=110,
        extra_edges=15,
        min_dist=9,
        seed=42
    )

    print("Map generated. Drawing graph...")
    draw_graph(G, positions)