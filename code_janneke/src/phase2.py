#phase 2: recourse with uncertain travel times

import math
import re
from pathlib import Path

import numpy as np
import vrplib
import VRPSolverEasy as vrpse


#heper functions

def tsplib_euc_2d(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(round(math.sqrt(dx * dx + dy * dy)))


def normalise_coords(coords):
    if hasattr(coords, "keys"):
        return {int(k): (float(v[0]), float(v[1])) for k, v in coords.items()}
    coords = np.asarray(coords)
    return {i + 1: (float(coords[i][0]), float(coords[i][1]))
            for i in range(coords.shape[0])}


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


def routes_to_arcs(routes):
    arcs = set()
    for r in routes:
        for a, b in zip(r[:-1], r[1:]):
            arcs.add((a, b))
    return arcs


#uncertainty model (correlated congestion travel times)

def sample_realised_times_correlated(
    base_times: np.ndarray,
    sigma_node: float = 0.15,
    sigma_edge: float = 0.05,
    seed: int = 0,
):
   
    if base_times.ndim != 2 or base_times.shape[0] != base_times.shape[1]:
        raise ValueError("base_times must be a square matrix (n x n).")

    rng = np.random.default_rng(seed)
    n = base_times.shape[0]

    #node-level noise (across row / column)
    row_noise = rng.normal(0.0, sigma_node, size=n)
    col_noise = rng.normal(0.0, sigma_node, size=n)

    row_factor = np.exp(row_noise)            #shape (n,), strictly > 0
    col_factor = np.exp(col_noise)            #shape (n,), strictly > 0

    #arc-level residual noise (independent for every arc)
    edge_noise = rng.normal(0.0, sigma_edge, size=(n, n))
    edge_factor = np.exp(edge_noise)          # shape (n, n), strictly > 0

    #combined multiplicative factor matrix
    factor = row_factor[:, None] * col_factor[None, :] * edge_factor

    #congestion-only, no speed-ups
    factor = np.maximum(1.0, factor)

    realised = base_times * factor

    #keep diagonal at 0 (no self travel)
    np.fill_diagonal(realised, 0.0)

    return realised


#solver

def solve_with_matrices(
    dists,
    times,
    demands,
    capacity,
    K,
    planned_arcs=None,
    lam_dev=0.0,
    tw_begin=0,
    tw_end=10**9,
):
    

    n = len(demands) - 1
    model = vrpse.Model()

    model.add_vehicle_type(
        id=1,
        start_point_id=0,
        end_point_id=0,
        name="V",
        capacity=int(capacity),
        max_number=int(K),
        var_cost_dist=1,
        tw_end=tw_end,
    )

    model.add_depot(id=0, name="DEPOT", tw_begin=tw_begin, tw_end=tw_end)

    for i in range(1, n + 1):
        model.add_customer(
            id=i,
            name=f"C{i}",
            demand=int(demands[i]),
            tw_begin=tw_begin,
            tw_end=tw_end,
        )

    for i in range(0, n + 1):
        for j in range(0, n + 1):
            if i == j:
                continue

            extra = 0.0
            if planned_arcs is not None and lam_dev > 0.0:
                if (i, j) not in planned_arcs:
                    extra = lam_dev

            model.add_link(
                i,
                j,
                distance=float(dists[i, j]) + extra,
                time=float(times[i, j]),
            )

    model.solve()
    routes = extract_routes(model.solution)
    return model, routes


#instance loading

def load_cvrplib_as_matrices(instance_path):
    data = vrplib.read_instance(instance_path)

    coords = normalise_coords(data["node_coord"])
    demands_raw = normalise_demands(data["demand"])
    capacity = int(data["capacity"])

    depot_node = data.get("depot", 1)
    if isinstance(depot_node, (list, tuple, np.ndarray)):
        depot_node = int(depot_node[0])
    if depot_node == 0:
        depot_node = 1

    customers = sorted([n for n in coords if n != depot_node])

    node_to_point = {depot_node: 0}
    for pid, n in enumerate(customers, start=1):
        node_to_point[n] = pid

    n = len(customers)
    dists = np.zeros((n + 1, n + 1))
    for i_node, i_pid in node_to_point.items():
        for j_node, j_pid in node_to_point.items():
            if i_pid != j_pid:
                dists[i_pid, j_pid] = tsplib_euc_2d(coords[i_node], coords[j_node])

    #base times proportional to distance
    times = dists.copy()

    demands = np.zeros(n + 1, dtype=int)
    for node, pid in node_to_point.items():
        if pid != 0:
            demands[pid] = demands_raw[node]

    return dists, times, demands, capacity


#main

if __name__ == "__main__":

    root = Path(__file__).resolve().parents[1]
    instance = root / "instances" / "A-n32-k5.vrp"

    #phase 1
    dists, base_times, demands, capacity = load_cvrplib_as_matrices(instance)

    model1, routes1 = solve_with_matrices(dists, base_times, demands, capacity, K=5)
    planned_arcs = routes_to_arcs(routes1)

    print("PHASE 1 routes:")
    print(routes1)

    #phase 2
    realised_times = sample_realised_times_correlated(
        base_times,
        sigma_node=0.15,   #shared congestion node-level (covariance)
        sigma_edge=0.05,   #independent arc-level noise
        seed=42,
    )
    mask = base_times > 0
    mult = np.ones_like(base_times, dtype=float)
    mult[mask] = realised_times[mask] / base_times[mask]

    #row/col mean multipliers - check!!
    row_mean = np.array([mult[i, mask[i]].mean() for i in range(mult.shape[0])])
    col_mean = np.array([mult[mask[:, j], j].mean() for j in range(mult.shape[1])])
    print("Row mean multipliers: min/median/max =", row_mean.min(), np.median(row_mean), row_mean.max())
    print("Col mean multipliers: min/median/max =", col_mean.min(), np.median(col_mean), col_mean.max())

    #multipliers stats of congestion - check!!
    mask = base_times > 0
    mult = realised_times[mask] / base_times[mask]
    print("\nCongestion multipliers stats:")
    print(f"  min={mult.min():.3f}  p05={np.quantile(mult, 0.05):.3f}  "
          f"median={np.median(mult):.3f}  p95={np.quantile(mult, 0.95):.3f}  max={mult.max():.3f}")

    #deviation penalty magnitude
    nonzero = dists[dists > 0]
    lam = 0.5 * float(np.median(nonzero))

    model2, routes2 = solve_with_matrices(
        dists,
        realised_times,
        demands,
        capacity,
        K=5,
        planned_arcs=planned_arcs,
        lam_dev=lam,
    )

    arcs2 = routes_to_arcs(routes2)
    deviating = len([a for a in arcs2 if a not in planned_arcs])

    objective = model2.solution.value

    print("\nPHASE 2 routes:")
    print(routes2)
    print("Deviating arcs:", deviating)
    print("Objective (distance + deviation penalty):", objective)
