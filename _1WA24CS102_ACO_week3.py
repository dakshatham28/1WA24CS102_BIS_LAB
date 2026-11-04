import numpy as np
import random
import matplotlib.pyplot as plt

# ==============================
# FINANCIAL LOCATIONS (CITIES)
# ==============================

cities = {
    "HeadOffice": (10, 20),
    "BranchA": (30, 40),
    "BranchB": (45, 10),
    "Client1": (60, 30),
    "Client2": (70, 50),
    "ATM1": (20, 60)
}

city_names = list(cities.keys())
coordinates = np.array(list(cities.values()))

n_cities = len(coordinates)

# ==============================
# DISTANCE MATRIX
# ==============================

distance_matrix = np.zeros((n_cities, n_cities))

for i in range(n_cities):
    for j in range(n_cities):
        distance_matrix[i][j] = np.linalg.norm(
            coordinates[i] - coordinates[j]
        )

# ==============================
# ACO PARAMETERS
# ==============================

n_ants = 20
n_iterations = 100

alpha = 1      # Pheromone importance
beta = 5       # Heuristic importance
rho = 0.5      # Evaporation rate
Q = 100        # Pheromone constant

pheromone = np.ones((n_cities, n_cities))

# ==============================
# TOUR DISTANCE FUNCTION
# ==============================

def calculate_distance(tour):
    distance = 0

    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i]][tour[i + 1]]

    distance += distance_matrix[tour[-1]][tour[0]]

    return distance

# ==============================
# SELECT NEXT CITY
# ==============================

def select_next_city(current_city, unvisited):

    probabilities = []

    for city in unvisited:

        pheromone_value = pheromone[current_city][city] ** alpha

        heuristic_value = (
            1 / distance_matrix[current_city][city]
        ) ** beta

        probabilities.append(
            pheromone_value * heuristic_value
        )

    probabilities = np.array(probabilities)

    probabilities = probabilities / probabilities.sum()

    next_city = np.random.choice(
        unvisited,
        p=probabilities
    )

    return next_city

# ==============================
# ANT COLONY OPTIMIZATION
# ==============================

best_tour = None
best_distance = float('inf')

for iteration in range(n_iterations):

    all_tours = []
    all_distances = []

    for ant in range(n_ants):

        start_city = random.randint(
            0,
            n_cities - 1
        )

        tour = [start_city]

        unvisited = list(range(n_cities))
        unvisited.remove(start_city)

        while unvisited:

            current_city = tour[-1]

            next_city = select_next_city(
                current_city,
                unvisited
            )

            tour.append(next_city)

            unvisited.remove(next_city)

        total_distance = calculate_distance(tour)

        all_tours.append(tour)
        all_distances.append(total_distance)

        if total_distance < best_distance:
            best_distance = total_distance
            best_tour = tour.copy()

    # ==========================
    # PHEROMONE EVAPORATION
    # ==========================

    pheromone *= (1 - rho)

    # ==========================
    # PHEROMONE UPDATE
    # ==========================

    for tour, distance in zip(
        all_tours,
        all_distances
    ):

        pheromone_deposit = Q / distance

        for i in range(len(tour) - 1):

            city1 = tour[i]
            city2 = tour[i + 1]

            pheromone[city1][city2] += pheromone_deposit
            pheromone[city2][city1] += pheromone_deposit

        pheromone[tour[-1]][tour[0]] += pheromone_deposit
        pheromone[tour[0]][tour[-1]] += pheromone_deposit

    print(
        f"Iteration {iteration + 1:3d} "
        f"| Best Distance = {best_distance:.2f}"
    )

# ==============================
# DISPLAY BEST ROUTE
# ==============================

print("\n" + "=" * 50)
print("OPTIMAL FINANCE MANAGEMENT ROUTE")
print("=" * 50)

for city in best_tour:
    print(city_names[city], end=" -> ")

print(city_names[best_tour[0]])

print("\nMinimum Distance:", round(best_distance, 2))

# ==============================
# VISUALIZE ROUTE
# ==============================

route = best_tour + [best_tour[0]]

x_coords = [
    coordinates[city][0]
    for city in route
]

y_coords = [
    coordinates[city][1]
    for city in route
]

plt.figure(figsize=(10, 6))

plt.plot(
    x_coords,
    y_coords,
    'bo-',
    linewidth=2,
    markersize=8
)

for i, city in enumerate(city_names):
    plt.annotate(
        city,
        (coordinates[i][0], coordinates[i][1]),
        fontsize=10,
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.title(
    "Ant Colony Optimization for Finance Management",
    fontsize=14
)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

plt.grid(True)

plt.show()
