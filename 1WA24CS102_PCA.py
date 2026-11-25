import numpy as np

jobs = [5, 2, 8, 3, 6]
n = 10
T = 30

def fitness(order):
    return sum(order[i] * (i + 1) for i in range(len(order)))

cells = []

for i in range(n):
    cells.append(np.random.permutation(jobs))

best_solution = min(cells, key=fitness)
best_fitness = fitness(best_solution)

for t in range(T):

    new_cells = []

    for i in range(n):

        left = cells[(i - 1) % n]
        right = cells[(i + 1) % n]

        neighbors = [cells[i], left, right]
        best_neighbor = min(neighbors, key=fitness)

        new_cell = best_neighbor.copy()

        a, b = np.random.choice(len(jobs), 2, replace=False)
        new_cell[a], new_cell[b] = new_cell[b], new_cell[a]

        new_cells.append(new_cell)

    cells = new_cells

    current_best = min(cells, key=fitness)

    if fitness(current_best) < best_fitness:
        best_solution = current_best.copy()
        best_fitness = fitness(best_solution)

print("Optimal Job Sequence:", best_solution)
print("Minimum Completion Cost:", best_fitness)
print("DAKSHATHA M")
print("1WA24CS102")
