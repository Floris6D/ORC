#phase1
import math
import re
from pathlib import Path

import numpy as np
import vrplib
import VRPSolverEasy as vrpse


def tsplib_euc_2d(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(round(math.sqrt(dx * dx + dy * dy)))


def normalise_coords(coords):
    #converts coords to dict: node_id -> (x, y)
    if hasattr(coords, "keys"):  # dict
        return {int(k): (float(v[0]), float(v[1])) for k, v in coords.items()}
    coords = np.asarray(coords)
    return {i + 1: (float(coords[i][0]), float(coords[i][1])) for i in range(coords.shape[0])}


def normalise_demands(demands):
    if hasattr(demands, "keys"):
        return {int(k): int(v) for k, v in demands.items()}
    demands = np.asarray(demands).reshape(-1)
    return {i + 1: int(demands[i]) for i in range(demands.shape[0])}


def extract_routes(solution):
    routes = []
    for line in str(solution).splitlines():
        if line.strip().startswith("ID :"):
            routes.append([int(x) for x in re.findall(r"\d+", line)])
    return routes


#phase1 solve

def solve_phase1(instance_path, K=5):
    data = vrplib.read_instance(instance_path)

    coords = normalise_coords(data["node_coord"])
    demands = normalise_demands(data["demand"])
    capacity = int(data["capacity"])

    depot_node = data.get("depot", 1)
    if isinstance(depot_node, (list, tuple, np.ndarray)):
        depot_node = int(depot_node[0])
    else:
        depot_node = int(depot_node)

    if depot_node == 0:
        depot_node = 1

    customers = [n for n in coords if n != depot_node]

    model = vrpse.Model()

    model.add_vehicle_type(
        id=1,
        start_point_id=0,
        end_point_id=0,
        name="V",
        capacity=capacity,
        max_number=K,
        var_cost_dist=1,
        tw_end=10**9
    )

    model.add_depot(id=0, name="DEPOT", tw_begin=0, tw_end=10**9)

    node_to_point = {depot_node: 0}
    for pid, n in enumerate(customers, start=1):
        node_to_point[n] = pid
        model.add_customer(
            id=pid,
            name=f"C{n}",
            demand=demands[n],
            tw_begin=0,
            tw_end=10**9
        )

    all_nodes = [depot_node] + customers
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            pi, pj = node_to_point[i], node_to_point[j]
            d = tsplib_euc_2d(coords[i], coords[j])
            model.add_link(pi, pj, distance=d, time=d)

    model.solve()
    routes = extract_routes(model.solution)

    return model, routes


#phase 1 results

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    instance = root / "instances" / "A-n32-k5.vrp"

    model, routes = solve_phase1(instance, K=5)

    print("Solved?", model.solution.is_defined())
    print("Routes (point IDs):", routes)
    print()
    print(model.solution)
    

