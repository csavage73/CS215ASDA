import random
import math
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tracemalloc




def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_spread_out_positions(num_nodes, width, height, min_dist, seed=random.randint(0,99999)):
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
    if random.randint(1, 100) < congestion_chance:
        return random.randint(1, 10)   # congested
    return random.randint(-10, 0)      # not congested


def generate_random_map(num_nodes=100, width=600, height=400, extra_edges=100, min_dist=5, seed=42, congestion_chance=30):
    positions = generate_spread_out_positions(num_nodes, width, height, min_dist, seed)
    G = nx.DiGraph()

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


def fix_negative_cycles(G):
    while nx.negative_edge_cycle(G, weight='weight'):
        # Initialize all distances to 0 so any negative cycle is reachable
        dist = {n: 0.0 for n in G.nodes}
        pred = {n: None for n in G.nodes}
        last_updated = None

        # Run |V| relaxations — last updated node is guaranteed inside a negative cycle
        for _ in range(len(G.nodes)):
            last_updated = None
            for u, v, data in G.edges(data=True):
                if dist[u] + data['weight'] < dist[v]:
                    dist[v] = dist[u] + data['weight']
                    pred[v] = u
                    last_updated = v

        if last_updated is None:
            break

        # Walk back |V| steps to ensure we land inside the cycle
        node = last_updated
        for _ in range(len(G.nodes)):
            node = pred[node]

        # Trace the cycle
        cycle_start = node
        cycle = []
        cur = node
        while True:
            cycle.append(cur)
            cur = pred[cur]
            if cur == cycle_start:
                break
        cycle.append(cycle_start)
        cycle.reverse()

        # Zero out the single most negative edge in the cycle to break it
        best_u, best_v, best_w = None, None, float('inf')
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            if G.has_edge(u, v):
                w = G[u][v]['weight']
                if w < best_w:
                    best_w, best_u, best_v = w, u, v

        if best_u is not None and best_w < 0:
            G[best_u][best_v]['weight'] = 0
        else:
            break


def nx_to_adj_dict(G):
    graph_dict = {}

    for node in G.nodes:
        graph_dict[node] = {}

    for u, v, data in G.edges(data=True):
        graph_dict[u][v] = data["weight"]

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
    visited = set()

    while current is not None:
        if current in visited:
            return None  # negative cycle detected
        visited.add(current)
        path.append(current)
        current = predecessors[current]

    path.reverse()
    '->'.join(path)

    if not path or path[0] != source:
        return None

    return '->'.join(path), path

def Analysis(lists, rounds, analysisType, unit=""):
    y = list(lists)
    x = list(range(rounds))
    mean_val = sum(y) / len(y)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color='steelblue', linewidth=1, alpha=0.6, label=f"Per-round {analysisType}")
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
               label=f"Mean: {mean_val:.6f} {unit}".strip())
    ax.set_title(f"{analysisType} per Iteration  (n={rounds})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(f"{analysisType} ({unit})" if unit else analysisType, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

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

    plt.title(f"Bellman-Ford Shortest Path: {source} → {target}  |  Total Cost = {cost}",
              fontsize=13, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(-5, max(x for x, _ in positions.values()) + 5)
    plt.ylim(-5, max(y for _, y in positions.values()) + 5)
    plt.tight_layout()


if __name__ == "__main__":
    CONGESTION_CHANCE = 23  # % chance an edge is congested (high weight). Change this value.
    NUM_NODES = 10         # Number of nodes in the graph
    rounds = 50         # Number of iterations. 
    # --- Static graph (same layout every run) ---
    #SEED = 42
    # --- Dynamic graph (different layout every run) --- uncomment the line below and comment out the line above
    SEED = random.randint(0, 999999)


    

    SOURCE = "N01"            #starting node
    TARGET = f"N{NUM_NODES:02}" #end node
    path = []
    times = []
    storage = []
    accuracy = []
    i = 0
    totalSum = 0
    
    while i < rounds:
        if i == rounds * 0.25:
            print("Loading 25%")
        elif i == rounds * 0.5:
            print("Loading 50%")
        elif i == rounds * 0.75:
            print("Loading 75%")

        G, positions = generate_random_map(
            num_nodes=NUM_NODES,
            width=600,
            height=400,
            extra_edges=100,
            min_dist=5,
            seed=SEED
        )

        fix_negative_cycles(G)
        graph_dict = nx_to_adj_dict(G)

        start_time = time.perf_counter()

        tracemalloc.stop()
        tracemalloc.start()

        fix_negative_cycles(G)
        graph_dict = nx_to_adj_dict(G)
        distances, predecessors = bellman_ford(graph_dict, SOURCE)
        result = reconstruct_path(predecessors, SOURCE, TARGET)
        shortest_path, path = result if result else (None, [])

        end_time = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        try:
            expected_preds, expected_dists = nx.bellman_ford_predecessor_and_distance(G, source=SOURCE)
            groundTest = distances == expected_dists
            if groundTest:
                accuracy.append(1)
            else:
                accuracy.append(0)

        except nx.NetworkXUnbounded:
            groundTest = None

        times.append(end_time - start_time)
        storage.append(peak)
        i += 1    
        
    totalAccuracy = (sum(accuracy) / rounds)
    timeTotalAve = (sum(times) / rounds)
    totalAveMem = sum(storage) / rounds
    print(f"================RESULTS==============================")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Source: {SOURCE}")
    print(f"Target: {TARGET}")
    print(f"Algorithm tested {rounds} times.")
    print("Total Average Time: ", timeTotalAve)
    print(f"Average Peak Memory: {totalAveMem / 1024:.2f} KB")
    print(f"Ground Test: {totalAccuracy:.2}%")
    print(f"Bellman-Ford Shortest Path: ", print(path) )
    print(f"Total Cost: {distances[TARGET]}")
    Analysis(times, rounds, "Time", unit="seconds")
    Analysis([p / 1024 for p in storage], rounds, "Memory", unit="KB")
    Analysis(accuracy, rounds, "Accuracy", unit="(1=pass, 0=fail)")

    draw_graph_with_path(G, positions, path, SOURCE, TARGET, distances[TARGET])
    plt.show()