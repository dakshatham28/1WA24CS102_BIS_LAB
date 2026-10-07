import random
import math

# ============================================================
# 1. TELECOMMUNICATION NETWORK
# ============================================================

network = {
    'A': [('B', 4), ('C', 6), ('D', 8)],
    'B': [('A', 4), ('C', 3), ('E', 5)],
    'C': [('A', 6), ('B', 3), ('D', 2), ('E', 4)],
    'D': [('A', 8), ('C', 2), ('E', 3), ('F', 6)],
    'E': [('B', 5), ('C', 4), ('D', 3), ('F', 2)],
    'F': [('D', 6), ('E', 2)]
}

SOURCE = 'A'
DESTINATION = 'F'


# ============================================================
# 2. GET COST BETWEEN TWO NODES
# ============================================================

def get_cost(node1, node2):

    for neighbor, cost in network[node1]:
        if neighbor == node2:
            return cost

    return float('inf')


# ============================================================
# 3. FIND ALL POSSIBLE ROUTES
# ============================================================

def find_all_paths(current, destination, path=None):

    if path is None:
        path = []

    path = path + [current]

    if current == destination:
        return [path]

    paths = []

    for neighbor, cost in network[current]:

        if neighbor not in path:

            new_paths = find_all_paths(
                neighbor,
                destination,
                path
            )

            paths.extend(new_paths)

    return paths


# Generate all possible routes
all_routes = find_all_paths(
    SOURCE,
    DESTINATION
)


# ============================================================
# 4. FITNESS FUNCTION
# ============================================================

def fitness(route):

    total_cost = 0

    for i in range(len(route) - 1):

        total_cost += get_cost(
            route[i],
            route[i + 1]
        )

    return total_cost


# ============================================================
# 5. PARTICLE CLASS
# ============================================================

class Particle:

    def __init__(self):

        # Random initial route
        self.position = random.choice(all_routes)

        # Initial velocity
        self.velocity = random.uniform(0.1, 0.5)

        # Personal Best
        self.pbest = self.position.copy()

        self.pbest_cost = fitness(
            self.position
        )


# ============================================================
# 6. PSO PARAMETERS
# ============================================================

NUM_PARTICLES = 5
MAX_ITERATIONS = 10

W = 0.7       # Inertia weight
C1 = 1.5      # Cognitive coefficient
C2 = 1.5      # Social coefficient


# ============================================================
# 7. DISPLAY AVAILABLE ROUTES
# ============================================================

print()
print("=" * 60)
print("             AVAILABLE NETWORK ROUTES")
print("=" * 60)

for i, route in enumerate(all_routes, 1):

    print(
        f"Route {i}: "
        f"{' -> '.join(route)} "
        f"| Cost = {fitness(route)}"
    )


# ============================================================
# 8. INITIALIZE PARTICLES
# ============================================================

particles = []

for _ in range(NUM_PARTICLES):

    particles.append(
        Particle()
    )


# ============================================================
# 9. FIND INITIAL GLOBAL BEST
# ============================================================

gbest_particle = min(
    particles,
    key=lambda p: p.pbest_cost
)

gbest = gbest_particle.pbest.copy()

gbest_cost = gbest_particle.pbest_cost


# ============================================================
# 10. DISPLAY INITIAL PARTICLES
# ============================================================

print()
print("=" * 60)
print("             INITIAL PARTICLE POPULATION")
print("=" * 60)

for i, particle in enumerate(particles, 1):

    print(
        f"P{i}: "
        f"Current = {' -> '.join(particle.position)}, "
        f"Cost = {fitness(particle.position)}, "
        f"pBest = {' -> '.join(particle.pbest)}, "
        f"pBest Cost = {particle.pbest_cost}"
    )


print()
print("Initial Global Best:")
print(
    f"gBest Route = {' -> '.join(gbest)}"
)
print(
    f"gBest Cost  = {gbest_cost}"
)


# ============================================================
# 11. PSO ITERATIONS
# ============================================================

for iteration in range(1, MAX_ITERATIONS + 1):

    print()
    print("=" * 60)
    print(f"                    ITERATION {iteration}")
    print("=" * 60)

    for i, particle in enumerate(particles, 1):

        # ----------------------------------------------------
        # CURRENT FITNESS
        # ----------------------------------------------------

        current_cost = fitness(
            particle.position
        )


        # ----------------------------------------------------
        # UPDATE PERSONAL BEST
        # ----------------------------------------------------

        if current_cost < particle.pbest_cost:

            particle.pbest = (
                particle.position.copy()
            )

            particle.pbest_cost = current_cost


        # ----------------------------------------------------
        # UPDATE GLOBAL BEST
        # ----------------------------------------------------

        if particle.pbest_cost < gbest_cost:

            gbest = (
                particle.pbest.copy()
            )

            gbest_cost = (
                particle.pbest_cost
            )


        # ----------------------------------------------------
        # RANDOM VALUES
        # ----------------------------------------------------

        r1 = random.random()
        r2 = random.random()


        # ----------------------------------------------------
        # UPDATE VELOCITY
        # ----------------------------------------------------

        particle.velocity = (
            W * particle.velocity
            + C1 * r1
            + C2 * r2
        )


        # ----------------------------------------------------
        # CONVERT VELOCITY TO PROBABILITY
        # ----------------------------------------------------

        probability = 1 / (
            1 + math.exp(
                -particle.velocity
            )
        )


        # ----------------------------------------------------
        # UPDATE POSITION
        # ----------------------------------------------------

        if random.random() < probability:

            current_cost = fitness(
                particle.position
            )

            # Find routes better than current route
            better_routes = []

            for route in all_routes:

                route_cost = fitness(route)

                if route_cost < current_cost:
                    better_routes.append(route)


            # Select a better route if available
            if better_routes:

                particle.position = random.choice(
                    better_routes
                )

            else:

                # Move toward global best
                particle.position = gbest.copy()


        # ----------------------------------------------------
        # CALCULATE NEW FITNESS
        # ----------------------------------------------------

        new_cost = fitness(
            particle.position
        )


        # ----------------------------------------------------
        # UPDATE PERSONAL BEST AGAIN
        # ----------------------------------------------------

        if new_cost < particle.pbest_cost:

            particle.pbest = (
                particle.position.copy()
            )

            particle.pbest_cost = new_cost


        # ----------------------------------------------------
        # UPDATE GLOBAL BEST AGAIN
        # ----------------------------------------------------

        if particle.pbest_cost < gbest_cost:

            gbest = (
                particle.pbest.copy()
            )

            gbest_cost = (
                particle.pbest_cost
            )


        # ----------------------------------------------------
        # DISPLAY PARTICLE INFORMATION
        # ----------------------------------------------------

        print(
            f"P{i}: "
            f"Current = {' -> '.join(particle.position)}, "
            f"Cost = {new_cost}, "
            f"pBest = {' -> '.join(particle.pbest)}, "
            f"pBest Cost = {particle.pbest_cost}"
        )


    # --------------------------------------------------------
    # DISPLAY GLOBAL BEST
    # --------------------------------------------------------

    print()
    print("Global Best after iteration", iteration)

    print(
        f"gBest Route = {' -> '.join(gbest)}"
    )

    print(
        f"gBest Cost  = {gbest_cost}"
    )


# ============================================================
# 12. FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("                  FINAL PSO RESULT")
print("=" * 60)

print(
    f"Source          : {SOURCE}"
)

print(
    f"Destination     : {DESTINATION}"
)

print(
    f"Optimal Route   : {' -> '.join(gbest)}"
)

print(
    f"Minimum Cost    : {gbest_cost}"
)

print(
    f"Iterations      : {MAX_ITERATIONS}"
)

print()
print("PSO successfully completed the network routing optimization.")
