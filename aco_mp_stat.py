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
    start = time.time()

    pheromone = np.ones(distances.shape) / len(distances)
    best_route = None
    best_length = float('inf')
    best_iter = 0

    iters = []

    for iter in range(n_iterations):
        all_routes = simulate_ants(n_ants, distances, pheromone, alpha, beta)
        all_routes.sort(key=lambda x: x[1])
        best_routes = all_routes[:n_best]
        update_pheromone(pheromone, best_routes, decay)
        if best_routes[0][1] < best_length:
            best_length = best_routes[0][1]
            best_route = best_routes[0][0]
            best_iter = iter
        iters.append((iter, best_routes[0][0], best_routes[0][1]))

    return best_iter, best_route, best_length, iters, start, time.time() - start


# Parameters for the ACO
n_ants = 59
n_best = 5
n_iterations = 300
decay = 0.1
alpha = 1
beta = 2


def worker_task(pid, start_event, terminate_event, queue, distances):
    start_event.wait()
    while not terminate_event.is_set():
        best_iter, best_route, best_length, iters, start, elapsed = ant_colony_optimization(distances, n_ants, n_best, n_iterations, decay, alpha, beta)
        f = open(RF".\data\ACO\{pid}_{start}_{elapsed}.csv", "w")
        for iter, route, length in iters:
            f.write(f"{iter};{length};{route}\n")
        f.close()
        queue.put((pid, start, elapsed, best_iter, best_route, best_length))


if __name__ == "__main__":

    file_path = R'.\dmat_Hamilton_szorg.xlsx'
    distance_matrix = load_matrix_from_excel(file_path)
    distances = distance_matrix.to_numpy()

    num_threads = os.cpu_count()
    queue = multiprocessing.Queue()
    start_event = multiprocessing.Event()
    terminate_event = multiprocessing.Event()
    processes = [multiprocessing.Process(target=worker_task, args=(pid, start_event, terminate_event, queue, distances))
                 for pid in range(num_threads)]

    for p in processes:
        p.start()

    start_time = time.time()

    start_event.set()

    best_distance = None
    best_route = None
    total_tries = 0

    f = open(R".\data\ACO\iterations.csv", "a")
    try:
        while total_tries < 1000:
            if not queue.empty():
                pid, start, elapsed, new_iter, new_route, new_distance = queue.get_nowait()
                f.write(f"{pid};{start};{elapsed};{new_iter};{new_distance};{new_route}\n")
                f.flush()
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
        f.close()

    print(F"Best Cost: {best_distance}, Best Path: {best_route}")
    print(F"Total tries: {total_tries}")
