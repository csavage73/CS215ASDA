import random
import math
import networkx as nx
import matplotlib.pyplot as plt


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_spread_out_positions(num_nodes, width, height, min_dist, seed=None):
    # seed=43        # uncomment and set a number for a reproducible layout
    # seed=None      # keep as None to use whatever seed is passed in (random each run)
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


def generate_random_map(num_nodes=100, width=200, height=130, extra_edges=15, min_dist=7, seed=random.randint(0,100)):  # change num_nodes default here if not passing it from __main__
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
        percent_value = 23
        chance = random.randint(0, 99)
        if chance < percent_value:
            congested = random.randint(5, 10)
            G.add_edge(u, v, weight=congested)
        else:
            not_Congested = random.randint(0, 5)
            G.add_edge(u, v, weight=not_Congested)
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
        percent_value = 23
        chance = random.randint(0, 99)
        if chance < percent_value:
            congested = random.randint(5,10)
            G.add_edge(u, v, weight=congested)
        else:
            not_Congested = random.randint(0,5)
            G.add_edge(u, v, weight=not_Congested)
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
                percent_value = 23
                chance = random.randint(0, 99)
                if chance < percent_value:
                    congested = random.randint(5,10)
                    G.add_edge(node, nearest, weight=congested)
                else:
                    not_Congested = random.randint(0,5)
                    G.add_edge(node, nearest, weight=not_Congested)
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


#PSO CONFIGURATION
NUM_PARTICLES = 40
MAX_ITERATIONS = 300
W  = 0.5   # inertia (probability of random mutation)
C1 = 1.5   # cognitive weight (pull toward personal best)
C2 = 2.0   # social weight    (pull toward global best)
# SOURCE and TARGET must match the first and last node labels for the given num_nodes.
# If num_nodes=50, change TARGET to "N50". If num_nodes=200, change TARGET to "N200".
SOURCE = "N01"
TARGET = "N100"  # change to f"N{num_nodes:02}" to match your node count


#PSO helpers

def path_weight(G, path):
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            return float("inf")
        total += G[u][v]["weight"]
    return total


def random_path(G, source, target, max_steps=2000):
    """Random walk from source to target (no revisits, backtracks on dead ends)."""
    visited = {source}
    path = [source]
    current = source
    for _ in range(max_steps):
        if current == target:
            return path
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            if len(path) == 1:
                return None
            path.pop()
            visited.discard(current)
            current = path[-1]
        else:
            nxt = random.choice(neighbors)
            path.append(nxt)
            visited.add(nxt)
            current = nxt
    return None


def crossover(path_a, path_b, source, target, G):
    """Splice a segment of path_b into path_a at a shared intermediate node."""
    shared = [n for n in path_a if n in path_b and n not in (source, target)]
    if not shared:
        return path_a[:]
    node = random.choice(shared)
    idx_a = path_a.index(node)
    idx_b = path_b.index(node)
    new_path = path_a[:idx_a + 1] + path_b[idx_b + 1:]
    if len(new_path) != len(set(new_path)):
        return path_a[:]
    if new_path[0] != source or new_path[-1] != target:
        return path_a[:]
    for i in range(len(new_path) - 1):
        if not G.has_edge(new_path[i], new_path[i + 1]):
            return path_a[:]
    return new_path


def mutate(G, path):
    """Replace a random interior segment with a fresh random sub-path."""
    if len(path) < 3:
        return path[:]
    interior = list(range(1, len(path) - 1))
    if len(interior) < 2:
        return path[:]
    i, j = sorted(random.sample(interior, 2))
    sub = random_path(G, path[i], path[j], max_steps=500)
    if sub is None:
        return path[:]
    new_path = path[:i] + sub + path[j + 1:]
    if len(new_path) != len(set(new_path)):
        return path[:]
    return new_path


#particle class

class Particle:
    def __init__(self, G, source, target):
        path = None
        while path is None:
            path = random_path(G, source, target)
        self.path = path
        self.fitness = path_weight(G, path)
        self.pbest_path = path[:]
        self.pbest_fitness = self.fitness

    def update(self, G, gbest_path, source, target):
        r1, r2 = random.random(), random.random()
        candidate = self.path[:]

        if r1 < C1 / (C1 + C2):
            candidate = crossover(candidate, self.pbest_path, source, target, G)

        if r2 < C2 / (C1 + C2):
            candidate = crossover(candidate, gbest_path, source, target, G)

        if random.random() < W:
            candidate = mutate(G, candidate)

        new_fitness = path_weight(G, candidate)
        if new_fitness < self.fitness:
            self.path = candidate
            self.fitness = new_fitness

        if self.fitness < self.pbest_fitness:
            self.pbest_path = self.path[:]
            self.pbest_fitness = self.fitness


#Main loop for the PSO
def run_pso(G, source=SOURCE, target=TARGET):
    dijkstra_path = nx.dijkstra_path(G, source, target, weight="weight")
    dijkstra_cost = path_weight(G, dijkstra_path)
    print(f"Dijkstra baseline : cost={dijkstra_cost}, nodes={len(dijkstra_path)}")

    swarm = [Particle(G, source, target) for _ in range(NUM_PARTICLES)]

    gbest_path = dijkstra_path[:]
    gbest_fitness = dijkstra_cost

    for p in swarm:
        if p.fitness < gbest_fitness:
            gbest_fitness = p.fitness
            gbest_path = p.path[:]

    history = [gbest_fitness]
    converged_at = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        for p in swarm:
            p.update(G, gbest_path, source, target)
            if p.pbest_fitness < gbest_fitness:
                gbest_fitness = p.pbest_fitness
                gbest_path = p.pbest_path[:]
                converged_at = iteration
        history.append(gbest_fitness)
        if iteration % 50 == 0:
            print(f"  Iter {iteration:>3}: best cost = {gbest_fitness}")

    print(f"\nPSO result        : cost={gbest_fitness}, nodes={len(gbest_path)}")
    print(f"Total iterations  : {MAX_ITERATIONS}")
    if converged_at is None:
        print(f"Converged at iter : N/A — PSO never improved on Dijkstra's solution")
    else:
        print(f"Converged at iter : {converged_at} / {MAX_ITERATIONS} ({100 * converged_at // MAX_ITERATIONS}% of budget used)")
    print(f"Path: {' -> '.join(gbest_path)}")
    return gbest_path, gbest_fitness, history


#visualization of the PSO

def draw_pso_result(G, positions, pso_path, dijkstra_path, history):
    _, axes = plt.subplots(1, 2, figsize=(20, 9))

    ax = axes[0]
    nx.draw_networkx_nodes(G, positions, node_size=200, node_color="steelblue", ax=ax)
    nx.draw_networkx_labels(G, positions, font_color="white", font_size=6, ax=ax)
    nx.draw_networkx_edges(G, positions, alpha=0.3, width=1, ax=ax)

    def path_edges(path):
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    nx.draw_networkx_edges(G, positions, edgelist=path_edges(dijkstra_path),
                           edge_color="orange", width=3, label="Dijkstra", ax=ax)
    nx.draw_networkx_edges(G, positions, edgelist=path_edges(pso_path),
                           edge_color="limegreen", width=2.5, style="dashed",
                           label="PSO", ax=ax)
    nx.draw_networkx_nodes(G, positions, nodelist=[SOURCE, TARGET],
                           node_size=500, node_color="red", ax=ax)

    for u, v, data in G.edges(data=True):
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        ox = (-dy / length * 1.2) if length != 0 else 0
        oy = (dx / length * 1.2) if length != 0 else 0
        ax.text(mx + ox, my + oy, str(data["weight"]), fontsize=5,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    pso_cost = path_weight(G, pso_path)
    dijk_cost = path_weight(G, dijkstra_path)
    ax.set_title(f"Shortest Path  |  PSO: {pso_cost}  |  Dijkstra: {dijk_cost}", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(history, color="steelblue", linewidth=1.5)
    axes[1].axhline(dijk_cost, color="orange", linestyle="--",
                    linewidth=1.5, label="Dijkstra optimal")
    axes[1].set_title("PSO Convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best Path Cost")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Generating random map...")
    G, positions = generate_random_map(
        num_nodes=100,   # CHANGE NODE COUNT HERE — also update TARGET above to match (e.g. num_nodes=50 -> TARGET="N50")
        width=170,       # increase width/height if nodes are too cramped after raising num_nodes
        height=110,
        extra_edges=15,  # more nodes = more edges needed to keep the graph well-connected
        min_dist=9,      # lower min_dist if placement fails with higher num_nodes
        # seed=43,                    # uncomment and set a number for a reproducible map
        seed=random.randint(0, 9999)  # random each run
    )

    print("Map generated. Running PSO...\n")
    pso_path, pso_cost, history = run_pso(G, SOURCE, TARGET)

    dijkstra_path = nx.dijkstra_path(G, SOURCE, TARGET, weight="weight")
    draw_pso_result(G, positions, pso_path, dijkstra_path, history)