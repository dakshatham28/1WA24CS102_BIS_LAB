import random

# ============================================================
# 1. DEFINE THE TELECOM NETWORK
# ============================================================

# Each link contains:
# latency  -> delay in milliseconds
# congestion -> value between 0 and 1
# cost -> monetary/operational cost

network = {
    'A': {
        'B': {'latency': 10, 'congestion': 0.2, 'cost': 5},
        'C': {'latency': 5,  'congestion': 0.3, 'cost': 3}
    },

    'B': {
        'A': {'latency': 10, 'congestion': 0.2, 'cost': 5},
        'C': {'latency': 7,  'congestion': 0.4, 'cost': 4},
        'D': {'latency': 4,  'congestion': 0.3, 'cost': 2},
        'E': {'latency': 8,  'congestion': 0.5, 'cost': 5}
    },

    'C': {
        'A': {'latency': 5, 'congestion': 0.3, 'cost': 3},
        'B': {'latency': 7, 'congestion': 0.4, 'cost': 4},
        'D': {'latency': 3, 'congestion': 0.2, 'cost': 2}
    },

    'D': {
        'B': {'latency': 4, 'congestion': 0.3, 'cost': 2},
        'C': {'latency': 3, 'congestion': 0.2, 'cost': 2},
        'E': {'latency': 6, 'congestion': 0.4, 'cost': 4}
    },

    'E': {
        'B': {'latency': 8, 'congestion': 0.5, 'cost': 5},
        'D': {'latency': 6, 'congestion': 0.4, 'cost': 4}
    }
}


SOURCE = 'A'
DESTINATION = 'E'


# ============================================================
# 2. INITIALIZE GA PARAMETERS
# ============================================================

POPULATION_SIZE = 30
GENERATIONS = 100

MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.80

ELITE_SIZE = 2


# ============================================================
# 3. CREATE INITIAL POPULATION
# ============================================================

def generate_random_path(source, destination):
    """
    Generate a random valid path from source to destination.
    Avoids visiting the same node twice.
    """

    path = [source]
    current = source
    visited = set(path)

    while current != destination:

        possible_nodes = [
            node for node in network[current]
            if node not in visited
        ]

        if not possible_nodes:
            return None

        # Give preference to destination if it is directly reachable
        if destination in possible_nodes and random.random() < 0.5:
            next_node = destination
        else:
            next_node = random.choice(possible_nodes)

        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


def create_initial_population():
    population = []

    while len(population) < POPULATION_SIZE:

        path = generate_random_path(SOURCE, DESTINATION)

        if path is not None:
            population.append(path)

    return population


# ============================================================
# 4. EVALUATE FITNESS
# ============================================================

def calculate_path_cost(path):
    """
    Calculate total network cost.

    Cost =
        latency
        + congestion penalty
        + operational cost
    """

    total_latency = 0
    total_congestion = 0
    total_cost = 0

    for i in range(len(path) - 1):

        current = path[i]
        next_node = path[i + 1]

        link = network[current][next_node]

        total_latency += link['latency']
        total_congestion += link['congestion']
        total_cost += link['cost']

    # Weights can be changed according to telecom requirements
    latency_weight = 1.0
    congestion_weight = 20.0
    cost_weight = 2.0

    total_cost_function = (
        latency_weight * total_latency
        + congestion_weight * total_congestion
        + cost_weight * total_cost
    )

    return total_cost_function


def fitness(path):
    """
    GA normally maximizes fitness.

    Since routing is a minimization problem,
    convert lower path cost into higher fitness.
    """

    path_cost = calculate_path_cost(path)

    return 1 / (1 + path_cost)


# ============================================================
# 5. SELECTION
# ============================================================

def selection(population):
    """
    Tournament selection.

    Randomly choose several individuals and
    select the one with the highest fitness.
    """

    tournament_size = 3

    tournament = random.sample(
        population,
        tournament_size
    )

    winner = max(
        tournament,
        key=fitness
    )

    return winner


# ============================================================
# 6. CROSSOVER
# ============================================================

def crossover(parent1, parent2):

    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    # Find common nodes
    common_nodes = list(
        set(parent1[1:-1]) &
        set(parent2[1:-1])
    )

    if not common_nodes:
        return parent1.copy(), parent2.copy()

    crossover_node = random.choice(common_nodes)

    index1 = parent1.index(crossover_node)
    index2 = parent2.index(crossover_node)

    child1 = parent1[:index1] + parent2[index2:]
    child2 = parent2[:index2] + parent1[index1:]

    # Make sure paths are valid
    if is_valid_path(child1):
        result1 = child1
    else:
        result1 = parent1.copy()

    if is_valid_path(child2):
        result2 = child2
    else:
        result2 = parent2.copy()

    return result1, result2


# ============================================================
# 7. MUTATION
# ============================================================

def mutation(path):

    mutated_path = path.copy()

    if random.random() > MUTATION_RATE:
        return mutated_path

    # If path has intermediate nodes,
    # randomly replace one intermediate node.
    if len(mutated_path) > 2:

        position = random.randint(
            1,
            len(mutated_path) - 2
        )

        previous_node = mutated_path[position - 1]

        possible_nodes = [
            node for node in network[previous_node]
            if node != mutated_path[position]
        ]

        if possible_nodes:

            new_node = random.choice(possible_nodes)

            mutated_path[position] = new_node

            # Try to construct a valid path from new node
            # to destination.
            remaining_path = generate_random_path(
                new_node,
                DESTINATION
            )

            if remaining_path:

                mutated_path = (
                    mutated_path[:position]
                    + remaining_path
                )

    return mutated_path


# ============================================================
# PATH VALIDATION
# ============================================================

def is_valid_path(path):

    if not path:
        return False

    if path[0] != SOURCE:
        return False

    if path[-1] != DESTINATION:
        return False

    # Check for repeated nodes
    if len(path) != len(set(path)):
        return False

    # Check whether every link exists
    for i in range(len(path) - 1):

        current = path[i]
        next_node = path[i + 1]

        if next_node not in network[current]:
            return False

    return True


# ============================================================
# 8. GENETIC ALGORITHM
# ============================================================

def genetic_algorithm():

    population = create_initial_population()

    best_path = None
    best_cost = float('inf')

    for generation in range(GENERATIONS):

        # Evaluate population
        population = sorted(
            population,
            key=calculate_path_cost
        )

        current_best = population[0]
        current_cost = calculate_path_cost(current_best)

        # Store global best solution
        if current_cost < best_cost:

            best_path = current_best.copy()
            best_cost = current_cost

        # Print progress
        print(
            f"Generation {generation + 1:3d} | "
            f"Best Path: {current_best} | "
            f"Cost: {current_cost:.2f}"
        )

        # Elitism
        new_population = [
            individual.copy()
            for individual in population[:ELITE_SIZE]
        ]

        # Create remaining population
        while len(new_population) < POPULATION_SIZE:

            parent1 = selection(population)
            parent2 = selection(population)

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutation(child1)
            child2 = mutation(child2)

            if is_valid_path(child1):
                new_population.append(child1)

            if (
                len(new_population) < POPULATION_SIZE
                and is_valid_path(child2)
            ):
                new_population.append(child2)

        population = new_population

    return best_path, best_cost


# ============================================================
# 9. OUTPUT THE BEST SOLUTION
# ============================================================

best_path, best_cost = genetic_algorithm()

print("\n======================================")
print("OPTIMAL ROUTING RESULT")
print("======================================")

print("Source      :", SOURCE)
print("Destination :", DESTINATION)
print("Best Path   :", " -> ".join(best_path))
print("Total Cost  :", round(best_cost, 2))

print("\nLink Details:")

total_latency = 0
total_congestion = 0
total_link_cost = 0

for i in range(len(best_path) - 1):

    current = best_path[i]
    next_node = best_path[i + 1]

    link = network[current][next_node]

    print(
        f"{current} -> {next_node} | "
        f"Latency: {link['latency']} ms | "
        f"Congestion: {link['congestion']} | "
        f"Cost: {link['cost']}"
    )

    total_latency += link['latency']
    total_congestion += link['congestion']
    total_link_cost += link['cost']

print("\nTotal Latency     :", total_latency, "ms")
print("Total Congestion  :", total_congestion)
print("Total Link Cost   :", total_link_cost)
