import numpy as np
import random

# ---------------------------------------------------
# GRAPH (Adjacency Matrix) 0 = no link
# ---------------------------------------------------
graph = np.array([
    [0, 2, 0, 1, 0],
    [2, 0, 3, 2, 0],
    [0, 3, 0, 0, 2],
    [1, 2, 0, 0, 3],
    [0, 0, 2, 3, 0]
])

num_nodes = len(graph)

# ---------------------------------------------------
# ACO PARAMETERS
# ---------------------------------------------------
num_ants = 20
num_iterations = 50

alpha = 1.0     # pheromone weight
beta = 2.0      # distance weight
evaporation_rate = 0.5
pheromone_deposit = 1.0

# Pheromones must be FLOAT (important!)
pheromones = np.ones_like(graph, dtype=float)

# ---------------------------------------------------
# PROBABILITY CALCULATION
# ---------------------------------------------------
def calculate_probabilities(pher_row, dist_row, visited):
    probs = np.zeros(num_nodes)
    for j in range(num_nodes):
        if dist_row[j] > 0 and j not in visited:
            probs[j] = (pher_row[j] ** alpha) * ((1 / dist_row[j]) ** beta)

    total = probs.sum()
    if total == 0:
        return probs  # dead end
    return probs / total

# ---------------------------------------------------
# GENERATE PATH FOR A SINGLE ANT
# ---------------------------------------------------
def ant_path(start, end):
    path = [start]
    current = start

    while current != end:

        distances = graph[current]
        pher = pheromones[current]

        probs = calculate_probabilities(pher, distances, path)

        if probs.sum() == 0:
            return None  # no possible route -> dead ant

        next_node = np.random.choice(range(num_nodes), p=probs)
        path.append(next_node)
        current = next_node

    return path

# ---------------------------------------------------
# CALCULATE PATH LENGTH
# ---------------------------------------------------
def path_length(path):
    total = 0
    for i in range(len(path) - 1):
        total += graph[path[i]][path[i+1]]
    return total

# ---------------------------------------------------
# MAIN ACO ALGORITHM
# ---------------------------------------------------
def ant_colony(start, end):
    global pheromones

    best_path = None
    best_length = float("inf")

    for iteration in range(num_iterations):
        all_paths = []
        all_lengths = []

        # Step 1: Each ant finds a path
        for _ in range(num_ants):
            path = ant_path(start, end)
            if path:
                length = path_length(path)
                all_paths.append(path)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = path

        # Step 2: Evaporate pheromones
        pheromones *= (1 - evaporation_rate)

        # Step 3: Deposit pheromone on good paths
        for path, length in zip(all_paths, all_lengths):
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                pheromones[a][b] += pheromone_deposit / length
                pheromones[b][a] += pheromone_deposit / length

    return best_path, best_length

# ---------------------------------------------------
# RUN ACO
# ---------------------------------------------------
source = 0
destination = 4

best_path, best_length = ant_colony(source, destination)

print("\n=== ACO SHORTEST PATH ===")
print("Best Path:", best_path)
print("Shortest Distance:", best_length)
