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