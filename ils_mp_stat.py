import pandas as pd
import random
import multiprocessing
import os
import time


def total_distance(route, distances):
    total = 0
    for i in range(len(route) - 1):
        total += distances[route[i]][route[i + 1]]
    total += distances[route[-1]][route[0]]
    return total


def perturb_solution(route, perturbation_size=5):
    indices_to_perturb = random.sample(range(len(route)), perturbation_size)
    perturbed_route = route.copy()

    cities_to_perturb = [perturbed_route[i] for i in indices_to_perturb]
    random.shuffle(cities_to_perturb)

    for idx, city in zip(indices_to_perturb, cities_to_perturb):
        perturbed_route[idx] = city

    return perturbed_route

def local_search(route, distances):
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 2):
            for j in range(i + 2, len(route)):
                if j == len(route) - 1:
                    next_j = 0
                else:
                    next_j = j + 1

                if distances[route[i]][route[i + 1]] + distances[route[j]][route[next_j]] > \
                        distances[route[i]][route[j]] + distances[route[i + 1]][route[next_j]]:
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                    improved = True

def ils_tsp(distances, max_iterations=300, perturbation_size=5):
    start = time.time()

    num_cities = len(distances)

    current_route = list(range(1, num_cities))
    random.shuffle(current_route)
    best_route = [0] + current_route + [0]
    best_length = total_distance(best_route, distances)
    best_iter = 0

    iters = []

    for iter in range(max_iterations):
        local_search(current_route, distances)

        perturbed_route = perturb_solution(current_route, perturbation_size)

        if len(set(perturbed_route)) == len(perturbed_route):
            local_search(perturbed_route, distances)

            current_length = total_distance(perturbed_route, distances)

            if current_length < best_length:
                best_route = [0] + perturbed_route + [0]
                best_length = current_length
                best_iter = iter

            iters.append((iter, [0] + perturbed_route + [0], current_length))

        current_route = best_route[1:-1]

    return best_iter, best_route, best_length, iters, start, time.time() - start

def load_matrix_from_excel(file_path):
    df = pd.read_excel(file_path, index_col=0)
    return df


def worker_task(pid, start_event, terminate_event, queue, distances):
    start_event.wait()
    while not terminate_event.is_set():
        best_iter, best_route, best_length, iters, start, elapsed = ils_tsp(distances)
        f = open(RF".data\ILS\{pid}_{start}_{elapsed}.csv", "w")
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

    f = open(R".\data\ILS\iterations.csv", "a")
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
