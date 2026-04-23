import random
import math
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tracemalloc
from pyswarm import pso
from dijkstar import find_path

CONGESTION_CHANCE = 23 # % chance an edge is congested (high weight). Change this value.
NUM_NODES = 10       # Number of nodes in the graph
rounds = 50         # Number of iterations. 
# --- Static graph (same layout every run) ---
#SEED = 42
# --- Dynamic graph (different layout every run) --- uncomment the line below and comment out the line above
rand_seed = True

#PSO CONFIGURATION
NUM_PARTICLES = 40
MAX_ITERATIONS = 500
DIMENSION = 2
max_steps = 500

W  = 0.35   # inertia (probability of random mutation)
C1 = 1.5   # cognitive weight (pull toward personal best)
C2 = 2.0   # social weight    (pull toward global best)
# SOURCE and TARGET must match the first and last node labels for the given num_nodes.
# If num_nodes=50, change TARGET to "N50". If num_nodes=200, change TARGET to "N200".
SOURCE = "N01"
TARGET = f"N{NUM_NODES:02}"  # change to total nodes
searchSpace = []


# --- PSO Helpers ---

def path_weight(G, path):
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            return float("inf")
        total += G[u][v]["weight"]
    return total


def fast_initial_path(G, positions, source, target):
    # Instead of random choice, pick neighbors closer to target with high probability
    path = [source]
    current = source
    while current != target:
        neighbors = list(G.neighbors(current))
        if not neighbors: return None # Handle dead ends
        # Weight neighbors by proximity to target
        weights = [1 / (euclidean_distance(positions[n], positions[target]) + 1e-6) for n in neighbors]
        current = random.choices(neighbors, weights=weights)[0]
        if current in path: break # Simple loop prevention
        path.append(current)
    return path if path[-1] == target else None


#Since inertia can't be used for a 2D search space
def crossover(path_a, path_b, source, target, G):
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


#Since Inertia can't be used for a 2D search space Mutation is needed to keep exploration and avoid early convergence
def mutate(G, positions, path):
    if len(path) < 3:
        return path[:]
    interior = list(range(1, len(path) - 1))
    if len(interior) < 2:
        return path[:]
    i, j = sorted(random.sample(interior, 2))
    sub = fast_initial_path(G, positions, path[i], path[j])
    if sub is None:
        return path[:]
    new_path = path[:i] + sub + path[j + 1:]
    if len(new_path) != len(set(new_path)):
        return path[:]
    return new_path


# --- Particle Class ---

class Particle:
    def __init__(self, G, path):
        self.path = path
        self.fitness = path_weight(G, path)
        self.pbest_path = path[:]
        self.pbest_fitness = self.fitness

    def update_pbest(self, G):
        self.fitness = path_weight(G, self.path)
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_path = self.path[:]

    def update(self, G, positions, gbest_path, source, target):
        r1, r2 = random.random(), random.random()
        candidate = self.path[:]

        if r1 < C1 / (C1 + C2):
            candidate = crossover(candidate, self.pbest_path, source, target, G)

        if r2 < C2 / (C1 + C2):
            candidate = crossover(candidate, gbest_path, source, target, G)

        if random.random() < W:
            candidate = mutate(G, positions, candidate)

        new_fitness = path_weight(G, candidate)
        if new_fitness < self.fitness:
            self.path = candidate
            self.fitness = new_fitness

        self.update_pbest(G)


def initialize_swarm(G, positions, source, target, num_particles, searchSpace):
    swarm = []
    for _ in range(num_particles):
        path = None
        while path is None:
            path = fast_initial_path(G, positions, source, target)
        swarm.append(Particle(G, path))
    return swarm


def run_pso(G, positions, source=SOURCE, target=TARGET):
    #intialize swarm based on graph, Starting node, Ending node, Particle count of swarm, and list of edges
    swarm = initialize_swarm(G, positions, source, target, NUM_PARTICLES, searchSpace)

    gbest_fitness = float("inf")
    gbest_path = None

    #update swarm
    for p in swarm:
        if p.fitness < gbest_fitness:
            gbest_fitness = p.fitness
            gbest_path = p.path[:]

    history = [gbest_fitness]

    for iteration in range(1, MAX_ITERATIONS + 1):
        for p in swarm:
            p.update(G, positions, gbest_path, source, target)
            if p.pbest_fitness < gbest_fitness:
                gbest_fitness = p.pbest_fitness
                gbest_path = p.pbest_path[:]
        history.append(gbest_fitness)
        #if iteration % 50 == 0:
        #    print(f"  Iter {iteration:>3}: best cost = {gbest_fitness}")

    #print(f"\nPSO result  : cost={gbest_fitness}, nodes={len(gbest_path)}")
    #print(f"Path: {' -> '.join(gbest_path)}")
    sol_path = ' -> '.join(gbest_path)
    return gbest_path, gbest_fitness, history, sol_path


# --- Graph Generation & Analysis ---

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_spread_out_positions(num_nodes, width, height, min_dist, seed=None):
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


def generate_random_map(num_nodes=100, width=200, height=130, extra_edges=15, min_dist=7, seed=random.randint(0,99999), CONGESTION_CHANCE = 23):  # change num_nodes default here if not passing it from __main__
    positions = generate_spread_out_positions(num_nodes, width, height, min_dist, seed)
    G = nx.DiGraph()

    for node in positions:
        G.add_node(node)

    node_list = list(positions.keys())
    connected = {node_list[0]}
    unconnected = set(node_list[1:])

    src = "N01"
    tgt = f"N{num_nodes:02}"

    def is_direct_src_tgt(a, b):
        return {a, b} == {src, tgt}

    def add_weighted_edge(u, v):
        chance = random.randint(1, 100)
        if chance < CONGESTION_CHANCE:
            G.add_edge(u, v, weight=random.randint(5, 10))
        else:
            G.add_edge(u, v, weight=random.randint(0, 5))

    # Step 1: Force full connectivity
    while unconnected:
        best_pair = None
        best_dist = float("inf")

        for u in connected:
            for v in unconnected:
                if is_direct_src_tgt(u, v):
                    continue
                dist = euclidean_distance(positions[u], positions[v])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (u, v)

        if best_pair is None:
            # fallback: only the src-tgt pair remains unconnected
            best_pair = next(
                ((u, v) for u in connected for v in unconnected), None
            )

        u, v = best_pair
        add_weighted_edge(u, v)
        connected.add(v)
        unconnected.remove(v)

    # Step 2: Add extra nearby edges
    possible_edges = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            u = node_list[i]
            v = node_list[j]
            if not G.has_edge(u, v) and not is_direct_src_tgt(u, v):
                dist = euclidean_distance(positions[u], positions[v])
                possible_edges.append((dist, u, v))

    possible_edges.sort(key=lambda x: x[0])

    added = 0
    for dist, u, v in possible_edges:
        if added >= extra_edges:
            break
        add_weighted_edge(u, v)
        added += 1

    # Step 3: Ensure every node has at least degree 2
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes):
            while G.degree[node] < 2:
                candidates = [
                    (euclidean_distance(positions[node], positions[other]), other)
                    for other in node_list
                    if other != node and not G.has_edge(node, other)
                    and not is_direct_src_tgt(node, other)
                ]
                candidates.sort(key=lambda x: x[0])

                if not candidates:
                    break

                _, nearest = candidates[0]
                add_weighted_edge(node, nearest)
                changed = True

    return G, positions


def draw_graph(G, positions, path, source, target, cost, dij_path=None, dij_cost=None):
    plt.figure(figsize=(16, 10))

    nx.draw_networkx_edges(G, positions, alpha=0.6, width=1.5, style='solid')

    if dij_path and len(dij_path) > 1:
        dij_edges = list(zip(dij_path[:-1], dij_path[1:]))
        nx.draw_networkx_edges(G, positions, edgelist=dij_edges, width=3.5, edge_color="blue", alpha=0.7, style='dashed')

    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, positions, edgelist=path_edges, width=3.5, edge_color="red", alpha=0.7)

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

    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color="red", linewidth=3, label=f"PSO (cost={cost})")]
    if dij_path:
        legend_handles.append(Line2D([0], [0], color="blue", linewidth=3, label=f"Dijkstra (cost={dij_cost})"))
    plt.legend(handles=legend_handles, fontsize=11, loc="upper left")

    plt.title(f"PSO vs Dijkstra: {source} → {target}", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(-5, max(x for x, _ in positions.values()) + 5)
    plt.ylim(-5, max(y for _, y in positions.values()) + 5)
    plt.tight_layout()
    #plt.show()
    


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

if __name__ == "__main__":
    
    SOURCE = "N01"            #starting node
    TARGET = f"N{NUM_NODES:02}" #end node
    path = []
    times = []
    storage = []
    accuracy = []
    sol_path = []
    i = 0
    totalSum = 0

    while i < rounds:
        if rand_seed == True:
            SEED = random.randint(0,99999)
        tracking = (i/rounds) * 100
        print(f"Tracking Iteration: {tracking}")
        #if i/rounds == 0.25:
        #    print("Loading 25%")
        #elif i/rounds == 0.50:
        #    print("Loading 50%")
        #elif i/rounds == 0.75:
        #    print("Loading 75%")
        
        G, positions = generate_random_map(
            num_nodes = NUM_NODES,
            width=600,
            height=400,
            extra_edges=100,
            min_dist=5,
            seed=SEED
        )

        #Ground testing — Dijkstra finds the true optimal cost to compare PSO against
        dijkstra = find_path(G, SOURCE, TARGET, cost_func=lambda _u, _v, e, _pe: e['weight'])
        dij_cost = dijkstra.total_cost
        dij_path = list(dijkstra.nodes)
        #print(f"Dijkstra    : cost={ground_cost}, path={' -> '.join(dijkstra.nodes)}")


        #timer start
        start_timer = time.perf_counter()

        tracemalloc.stop()   # clear any prior state
        tracemalloc.start()  # fresh baseline = 0

        #run the PSO
        pso_path, pso_cost, history, sol_path = run_pso(G, positions, SOURCE, TARGET)

        #timer end
        end_timer = time.perf_counter()

        #get memory peak then stop so next iteration starts clean
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if pso_cost == dij_cost:
            accuracy.append(1)
        else:
            accuracy.append(0)
        
        #list times
        times.append(end_timer - start_timer)
        storage.append(peak)

        i += 1

    totalAccuracy = ((sum(accuracy) / rounds) * 100)
    timeTotalAve = (sum(times) / rounds)
    totalAveMem = sum(storage) / rounds
    #Final output to terminal
    print(f"================RESULTS==============================")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Source: {SOURCE}")
    print(f"Target: {TARGET}")
    print(f"Algorithm tested {rounds} times.")
    print(f"Solution Path for Final Test: {sol_path}")
    print("Total Average Time: ", timeTotalAve)
    print(f"Average Peak Memory: {totalAveMem / 1024:.2f} KB")
    print(f"Ground Test: {totalAccuracy}%")
    print(f"Total Cost: {pso_cost}")
    
    #Final output to window
    Analysis(times, rounds, "Time", unit="seconds")
    Analysis([p / 1024 for p in storage], rounds, "Memory", unit="KB")
    Analysis(accuracy, rounds, "Accuracy")
    draw_graph(G, positions, pso_path, SOURCE, TARGET, pso_cost, dij_path=dij_path, dij_cost=dij_cost)
    plt.show()
