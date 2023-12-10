import pandas as pd
import numpy as np
import multiprocessing
import os
import time


def load_matrix_from_excel(file_path):
    df = pd.read_excel(file_path, index_col=0)
    return df


def calculate_route_length(route, distances):
    return sum([distances[route[i-1]][route[i]] for i in range(len(route))])


def choose_next_city(current_city, unvisited, pheromone, distances, alpha, beta):
    pheromone_values = pheromone[current_city][unvisited]
    distances_values = distances[current_city][unvisited]
    pheromone_values = pheromone_values ** alpha
    heuristic_values = (1 / distances_values) ** beta
    probabilities = pheromone_values * heuristic_values
    probabilities /= probabilities.sum()
    return np.random.choice(unvisited, 1, p=probabilities)[0]


def simulate_ants(n_ants, distances, pheromone, alpha, beta):
    all_routes = []
    all_inds = range(len(distances))
    for _ in range(n_ants):
        route = [np.random.randint(len(distances))]
        unvisited = set(all_inds) - set(route)
        while unvisited:
            current_city = route[-1]
            next_city = choose_next_city(current_city, list(unvisited), pheromone, distances, alpha, beta)
            route.append(next_city)
            unvisited.remove(next_city)
        all_routes.append((route, calculate_route_length(route, distances)))
    return all_routes


def update_pheromone(pheromone, best_routes, decay):
    pheromone *= (1 - decay)
    for route, length in best_routes:
        for i in range(len(route)):
            pheromone[route[i-1]][route[i]] += 1.0 / length


def ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
    pheromone = np.ones(distances.shape) / len(distances)
    best_route = None
    best_length = float('inf')
    best_iter = 0

    for iter in range(n_iterations):
        all_routes = simulate_ants(n_ants, distances, pheromone, alpha, beta)
        all_routes.sort(key=lambda x: x[1])
        best_routes = all_routes[:n_best]
        update_pheromone(pheromone, best_routes, decay)
        if best_routes[0][1] < best_length:
            best_length = best_routes[0][1]
            best_route = best_routes[0][0]
            best_iter = iter

    return best_iter, best_route, best_length


def validate(dists, route, named_distance):
    if len(route) != len(dists[0]) + 1:
        print(f"Total distance invalid! Result city count:{len(route)}, expected:{len(dists[0])}")
        return
    calculated_dist = 0
    print(len(route))
    for i in range(len(route) - 1):
        current_dist = dists[route[i] - 1][route[i + 1] - 1]
        calculated_dist += current_dist
        print(f"dist({route[i]}, {route[i+1]}) = {current_dist}   \tTemp sum: {calculated_dist}")

    if named_distance == calculated_dist:
        print("Total distance valid!")
    else:
        print(f"Total distance invalid! Named:{named_distance}, actual:{calculated_dist} ")

def increase_city_idxs(route):
    return [x + 1 for x in route]


def offset_route_to_one(route):
    idx = route.index(1)
    return route[idx:] + route[:idx]


# Parameters for the ACO
n_ants = 59
n_best = 5
n_iterations = 300
decay = 0.1
alpha = 1
beta = 2


def worker_task(start_event, terminate_event, queue, distances):
    start_event.wait()
    while not terminate_event.is_set():
        queue.put(ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha, beta))


if __name__ == "__main__":

    file_path = input("Input the xlsx path: ")
    distance_matrix = load_matrix_from_excel(file_path)
    distances = distance_matrix.to_numpy()

    num_threads = os.cpu_count()
    queue = multiprocessing.Queue()
    start_event = multiprocessing.Event()
    terminate_event = multiprocessing.Event()
    processes = [multiprocessing.Process(target=worker_task, args=(start_event, terminate_event, queue, distances))
                 for _ in range(num_threads)]

    for p in processes:
        p.start()

    n_seconds = int(input("How many seconds can the algorithm run? "))

    start_time = time.time()

    start_event.set()

    best_distance = None
    best_route = None
    total_tries = 0

    try:
        while time.time() - start_time < n_seconds:
            if not queue.empty():
                new_iter, new_route, new_distance = queue.get_nowait()
                if best_distance is None or new_distance < best_distance:
                    best_distance = new_distance
                    best_route = offset_route_to_one(increase_city_idxs(new_route)) + [1]
                    print(F"Cost: {best_distance}, Path: {best_route}")
                total_tries += 1
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        terminate_event.set()
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

    print(F"Best Cost: {best_distance}, Best Path: {best_route}")
    print(F"Total tries: {total_tries}")

    validate(distances, best_route, best_distance)
