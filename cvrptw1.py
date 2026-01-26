#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:46:01 2026

@author: martijnkrikke
"""
import os
import matplotlib.pyplot as plt
import VRPSolverEasy as solver
import math
import numpy as np
from scipy.spatial.distance import cdist
from distance_mutator import generate_correlated_mutations
from constants import *
import logging
import time
import pandas as pd
import concurrent.futures
import logging
import time
import ast



def read_instance(filepath):
    """
    Leest een Solomon VRPTW file en returnt een dictionary met specifieke variabelen.
    """
    # Initialiseer opslag
    vehicle_info = {}
    customers_data = [] # Hier komen alle regels (depot + klanten) in volgorde
    
    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Sectie detectie
        if line.startswith("VEHICLE"):
            section = "vehicle_header"
            continue
        elif line.startswith("CUSTOMER"):
            section = "customer_header"
            continue
            
        # Data verwerking
        if section == "vehicle_header":
            if line.startswith("NUMBER"): section = "vehicle_data"
            
        elif section == "vehicle_data":
            parts = line.split()
            vehicle_info["number"] = int(parts[0])
            vehicle_info["capacity"] = int(parts[1])
            section = None
            
        elif section == "customer_header":
            if line.startswith("CUST"): section = "customer_data"
            
        elif section == "customer_data":
            # Formaat: CUST NO. X Y DEMAND READY DUE SERVICE
            parts = line.split()
            if len(parts) >= 7:
                # Sla de ruwe data tijdelijk op
                customers_data.append([int(float(p)) for p in parts])

    # --- Data verwerking naar gewenste lijsten ---
    
    # Regel 0 is het depot, de rest zijn klanten
    depot_raw = customers_data[0]
    cust_raw = customers_data[1:]
    
    # 1. Algemene variabelen
    vehicle_capacity = vehicle_info.get("capacity", 0)
    max_number = vehicle_info.get("number", 0)
    nb_customers = len(cust_raw)
    
    # 2. Depot variabelen
    depot_coordinates = [depot_raw[1], depot_raw[2]] # X, Y
    depot_service_time = depot_raw[6]
    
    # Let op: Standaard is 'Ready Time' het begin en 'Due Date' het eind
    depot_tw_begin = depot_raw[4] 
    depot_tw_end = depot_raw[5]
    
    # 3. Klant variabelen (Lijsten)
    cust_demands = [c[3] for c in cust_raw]
    cust_coordinates = [[c[1], c[2]] for c in cust_raw]
    cust_tw_begin = [c[4] for c in cust_raw]
    cust_tw_end = [c[5] for c in cust_raw]
    cust_service_time = [c[6] for c in cust_raw]

    # Retourneer alles in een dictionary
    return {
        "vehicle_capacity": vehicle_capacity,
        "max_number": max_number,
        "nb_customers": nb_customers,
        "cust_demands": cust_demands,
        "cust_coordinates": cust_coordinates,
        "depot_coordinates": depot_coordinates,
        "depot_service_time": depot_service_time,
        "depot_tw_end": depot_tw_end,
        "depot_tw_begin": depot_tw_begin,
        "cust_tw_begin": cust_tw_begin,
        "cust_tw_end": cust_tw_end,
        "cust_service_time": cust_service_time
    }

def get_data(path, total_customers=TOTAL_CUSTOMERS):
    if isinstance(path, str):
        if ".txt" not in path:
            path += ".txt"
        if "In/" not in path:
            path = "In/" + path
    data = read_instance(path)
    #Distance matrix berekenen
    all_coordinates = [data["depot_coordinates"]] + data["cust_coordinates"][:total_customers]
    coords_array = np.array(all_coordinates)
    distance_matrix = cdist(coords_array, coords_array)
    return data, distance_matrix

def compute_euclidean_distance(x_i, x_j, y_i, y_j):
  """compute the euclidean distance between 2 points from graph"""
  return round(math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2), 3)
 
def plot_solution(data, solution, total_customers, custom_cost = None):
    """
    Plot de routes van de oplossing bovenop de klantlocaties.
    """
    # 1. Basis: Plot alle pointen (Klanten en Depot)
    cust_x = [c[0] for c in data['cust_coordinates']]
    cust_y = [c[1] for c in data['cust_coordinates']]
    depot_x, depot_y = data['depot_coordinates']

    plt.figure(figsize=(10, 8))
    
    # Teken klanten (grijs/lichtblauw zodat de routes opvallen)
    plt.scatter(cust_x, cust_y, c='lightgray', edgecolors='black', s=30, label='Klanten')
    
    # Teken depot
    plt.scatter(depot_x, depot_y, c='black', marker='s', s=100, label='Depot', zorder=10)

    # 2. Routes extraheren en tekenen
    # We gebruiken een kleurencyclus zodat elke route een andere kleur heeft
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    # solution.routes is een lijst van route-objecten
    for i, route in enumerate(solution.routes):
        route_ids = route.point_ids # Dit geeft een lijst: [0, 20, 1, 6, ..., 0]
        
        x_coords = []
        y_coords = []
        
        for node_id in route_ids:
            if node_id == 0:
                # Depot coördinaten
                x_coords.append(depot_x)
                y_coords.append(depot_y)
            else:
                # Klant coördinaten
                # Let op: ID 1 in solver = index 0 in jouw data-lijst
                cust_idx = node_id - 1
                # Check of index bestaat (voor het geval je niet alle klanten in het model stopt)
                if 0 <= cust_idx < total_customers:
                    x_coords.append(data['cust_coordinates'][cust_idx][0])
                    y_coords.append(data['cust_coordinates'][cust_idx][1])
        
        # Kies een kleur
        color = colors[i % len(colors)]
        
        # Teken de lijn
        plt.plot(x_coords, y_coords, c=color, linewidth=2, label=f'Route {i+1}, id={route.vehicle_type_id}, #nodes={len(route_ids)}', alpha=0.8)
        
        # Optioneel: Pijltjes om de richting aan te geven (halverwege de lijnstukken)
        # Dit maakt de plot soms druk, dus je kunt dit weglaten als je wilt
        for k in range(len(x_coords) - 1):
            mid_x = (x_coords[k] + x_coords[k+1]) / 2
            mid_y = (y_coords[k] + y_coords[k+1]) / 2
            dx = (x_coords[k+1] - x_coords[k]) * 0.1 # kortere pijl
            dy = (y_coords[k+1] - y_coords[k]) * 0.1
            plt.arrow(mid_x, mid_y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=1.5, color=color)

    # Opmaak
    if custom_cost is not None:
        plt.title(f"Oplossing C201 (Kosten: {round(custom_cost, 2)})")
    else:
        plt.title(f"Oplossing C201 (Kosten: {round(solution.value, 2)})")
    plt.xlabel('X Coördinaat')
    plt.ylabel('Y Coördinaat')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    savename = 'plot_single'
    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")
    i = 0
    while os.path.exists(f"plots/{savename}_{i}.png"):
        i += 1
    plt.savefig(f"plots/{savename}_{i}.png")
    print("Plot saved to:", f"plots/{savename}_{i}.png")


def plot_two_solutions(data, solution1, solution2, total_customers, titles=None, proxy_extra_cost=0, manual_title = False):
    """
    Plot twee oplossingen naast elkaar bovenop de klantlocaties.
    """
    if titles is None:
        titles = ["Oplossing 1", "Oplossing 2"]
    
    cust_x = [c[0] for c in data['cust_coordinates']]
    cust_y = [c[1] for c in data['cust_coordinates']]
    depot_x, depot_y = data['depot_coordinates']

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    solutions = [solution1, solution2]

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    for ax, sol, title in zip(axes, solutions, titles):
        # Klanten
        ax.scatter(cust_x, cust_y, c='lightgray', edgecolors='black', s=30, label='Klanten')
        # Depot
        ax.scatter(depot_x, depot_y, c='black', marker='s', s=100, label='Depot', zorder=10)

        # Routes
        for i, route in enumerate(sol.routes):
            route_ids = route.point_ids
            x_coords = []
            y_coords = []
            for node_id in route_ids:
                if node_id == 0:
                    x_coords.append(depot_x)
                    y_coords.append(depot_y)
                else:
                    cust_idx = node_id - 1
                    if 0 <= cust_idx < total_customers:
                        x_coords.append(data['cust_coordinates'][cust_idx][0])
                        y_coords.append(data['cust_coordinates'][cust_idx][1])
            color = colors[i % len(colors)]
            ax.plot(x_coords, y_coords, c=color, linewidth=2, label=f'Route {i+1}, id={route.vehicle_type_id}, #nodes={len(route_ids)}', alpha=0.8)
            
            # Optionele pijltjes
            for k in range(len(x_coords) - 1):
                mid_x = (x_coords[k] + x_coords[k+1]) / 2
                mid_y = (y_coords[k] + y_coords[k+1]) / 2
                dx = (x_coords[k+1] - x_coords[k]) * 0.1
                dy = (y_coords[k+1] - y_coords[k]) * 0.1
                ax.arrow(mid_x, mid_y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=1.5, color=color)

        ax.set_title(f"{title} (Kosten: {round(sol.value, 2)})")
        if manual_title:
            ax.set_title(title)
        elif "phase 2" in title.lower() or "phase2" in title.lower():
            ax.set_title(f"{title} (Kosten: {round(sol.value - proxy_extra_cost, 2)} excl. proxy kosten)")
        ax.set_xlabel('X Coördinaat')
        ax.set_ylabel('Y Coördinaat')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    
    plt.tight_layout()
    savename = 'plot'
    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")
    i = 0
    while os.path.exists(f"plots/{savename}_{i}.png"):
        i += 1
    plt.savefig(f"plots/{savename}_{i}.png")
    print("Plot saved to:", f"plots/{savename}_{i}.png")

def increase_time_windows(data, increase_amount=100):
    """Increase all time windows by a fixed amount."""
    data["depot_tw_end"] += increase_amount
    data["depot_tw_begin"] = max(0, data["depot_tw_begin"] - increase_amount)
    data["cust_tw_begin"] = [max(0, twb - increase_amount) for twb in data["cust_tw_begin"]]
    data["cust_tw_end"] = [twe + increase_amount for twe in data["cust_tw_end"]]
    return data

def phase1(data, distance_matrix, total_customers=TOTAL_CUSTOMERS, alpha = 1, beta = 5, print_solution=True):
    # model. depot -> vehicle -> customer -> links er in
    # alpha is de 'zekerheids multiplier', dus alpha =1 betekend geen zekerheidsmarge
    model0 = solver.Model()


    model0.add_depot(id=0,
        service_time=data["depot_service_time"],
        tw_begin=data["depot_tw_begin"],
        tw_end=data["depot_tw_end"]
        )


    model0.add_vehicle_type(id=1,
        start_point_id=0,
        end_point_id=0,
        max_number=data["max_number"],
        capacity=data["vehicle_capacity"],
        tw_begin=data["depot_tw_begin"],
        tw_end=data["depot_tw_end"],
        var_cost_time=1,
        )

    for i in range(total_customers):
        model0.add_customer(
            id=i + 1,
            service_time=data["cust_service_time"][i],
            tw_begin=data["cust_tw_begin"][i],
            tw_end=data["cust_tw_end"][i],
            demand=data["cust_demands"][i]
        )    
        

    rows, cols = distance_matrix.shape


    for i in range(rows):
        for j in range(i + 1, cols):
            dist = distance_matrix[i][j] *alpha + beta
            model0.add_link(
                start_point_id=i, 
                end_point_id=j, 
                distance=dist, 
                time=dist
            )

    # model solven
    model0.set_parameters(time_limit=10000, solver_name="CLP")
    model0.solve()
    if print_solution:
        print(model0.solution)
        # plot_solution(data, model0.solution, total_customers=total_customers)
    

    return model0

def phase2(data, distance_matrix, model0, total_customers=TOTAL_CUSTOMERS, print_solution=True, noise_matrix=None, noise_params=NOISE_PARAMS):
    ################Fase twee
    #routes en loads uit de solution halen
    route = model0.solution.routes
    number_of_trucks = len(route)
    load_per_truck = []
    for i in range(number_of_trucks):
        value = model0.solution.routes[i].cap_consumption[-1]
        load_per_truck.append(int(value))
    
    np.random.seed(0)
    if noise_matrix is None:
        noise_matrix = generate_correlated_mutations(distance_matrix.shape[0], noise_params)
    distance_matrix_stochastic = distance_matrix * noise_matrix

    ### Tweede model beginnen
    model1 = solver.Model()

    model1.add_depot(id=0,
        service_time=data["depot_service_time"],
        tw_begin=data["depot_tw_begin"],
        tw_end=data["depot_tw_end"]
        )

    #Adding both the original trucks, and the repalcement trucks
    #tracking which original truck links to which proxy truck, used later to ensure that at least one will be chosen
    #both get VERY_HIGH_COST to ensure that the optimal solution never uses both
    
    truck_link = {}
    for i, capacity in enumerate(load_per_truck):
        #Original trucks
        og_truck_id = i+1
        model1.add_vehicle_type(
            id=og_truck_id,
            start_point_id=0,
            end_point_id=0,
            max_number=1,
            capacity=capacity+1,  #+1 for going through the XOR node with demand =1
            tw_begin=data["depot_tw_begin"],
            tw_end=data["depot_tw_end"],
            var_cost_time=1, 
            fixed_cost=VERY_HIGH_COST,
            name=f"OG truck {i+1}"
        )

        #Proxy trucks
        proxy_truck_id  = og_truck_id + 100
        model1.add_vehicle_type(
            id=proxy_truck_id,
            start_point_id=0,
            end_point_id=0,
            max_number=1,
            capacity=data["vehicle_capacity"]+1, #+1 for going through the XOR node with demand=1
            tw_begin=data["depot_tw_begin"],
            tw_end=data["depot_tw_end"],
            var_cost_time=1,
            fixed_cost=FIXED_COST_FOR_RELOADING + VERY_HIGH_COST,
            name=f"Proxy truck {i+1}"
        )

        truck_link[og_truck_id] = proxy_truck_id


    #add the real customers
    for i in range(total_customers):
        model1.add_customer(
            id=i + 1,
            service_time=data["cust_service_time"][i],
            tw_begin=data["cust_tw_begin"][i],
            tw_end=data["cust_tw_end"][i],
            demand=data["cust_demands"][i]
        )    

    rows, cols = distance_matrix_stochastic.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            dist = distance_matrix_stochastic[i][j]
            model1.add_link(
                start_point_id=i, 
                end_point_id=j, 
                distance=dist, 
                time=dist
            )

    # Add nodes where either original truck or proxy truck HAS to go to, 
    # and only these truck types are allowed there
    # Essentially an XOR gate (exactly one, but not both)
    # Boom, very cool
    all_vehicle_types = list(truck_link.keys()) + list(truck_link.values())
    
    for i, truck_OG in enumerate(truck_link):
        truck_proxy = truck_link[truck_OG]
        xor_node_id = total_customers + 1 + i
        model1.add_customer(
            id=xor_node_id,
            service_time=0,
            tw_begin=data["depot_tw_begin"],
            tw_end=data["depot_tw_end"],
            demand=1,
            incompatible_vehicles = [vt for vt in all_vehicle_types if vt not in [truck_OG, truck_proxy]],
        )
        
        model1.add_link(
            start_point_id=0,
            end_point_id=xor_node_id,
            distance=0,
            time=0,
        )

        for customer in range(1, total_customers + 1):
            dist = distance_matrix_stochastic[0][customer]
            model1.add_link(
                start_point_id=xor_node_id,
                end_point_id=customer,
                distance=0,
                time=0,
            )

    # later needed for calculating actual cost, as the cost of the model is changed now
    proxy_extra_cost = len(truck_link) * (VERY_HIGH_COST)
    # len(truck_link) * VERY_HIGH_COST for the fixed costs of the proxy trucks

    # model solven en plotten
    model1.set_parameters(time_limit=10000, solver_name="CLP")
    model1.solve()
    if print_solution:
        print(model1.solution)
        # plot_solution(data, model1.solution, total_customers=total_customers)
    return model1, proxy_extra_cost

def run_instance(path,
                  alpha, 
                  beta,
                  total_customers=TOTAL_CUSTOMERS,  
                  noise_params=NOISE_PARAMS, 
                  print_solution=False,
                  plot_both_solutions=False,
                  increase_tw_amount=0,
                  seed=0,
                  only_return_cost=False):
    #path can either be of form 'c201' or 'In/c201.txt'
    data, distance_matrix = get_data(path, total_customers=total_customers)
    data = increase_time_windows(data, increase_amount=increase_tw_amount)
    noise_matrix, distance_matrix_stochastic = build_phase2_stochastic_matrix(
        distance_matrix=distance_matrix,
        noise_params=noise_params,
        seed=seed,
    )

    model0 = phase1(data = data,
                    distance_matrix = distance_matrix, 
                    alpha = alpha, 
                    beta = beta,
                    total_customers=total_customers, 
                    print_solution=print_solution)

    model1, proxy_extra_cost = phase2(data = data,
                    distance_matrix = distance_matrix, 
                    model0 = model0, 
                    total_customers=total_customers, 
                    print_solution=print_solution, 
                    noise_matrix=noise_matrix)
    
    cost = compute_routing_cost_from_matrix(model1.solution, distance_matrix_stochastic, max_real_node_id=total_customers) + count_proxy_trucks_used(model1.solution) * FIXED_COST_FOR_RELOADING

    
    if plot_both_solutions:
        plot_two_solutions(data, model0.solution, model1.solution, total_customers=total_customers,  titles=["Phase 1", "Phase 2"], proxy_extra_cost=proxy_extra_cost)
    if only_return_cost:
        return cost
    return model0, model1

def _test_run_multiple_params_phase1(path,
                  params, 
                  total_customers=TOTAL_CUSTOMERS):
    #path can either be of form 'c201' or 'In/c201.txt'
    data, distance_matrix = get_data(path, total_customers=total_customers)
    data = increase_time_windows(data, increase_amount=100)
    models = []
    for param in params:
        alpha = param[0]
        beta = param[1]
        model0 = phase1(data, distance_matrix, 
                        alpha=alpha, 
                        beta=beta,
                        total_customers=total_customers, 
                        print_solution=False)
        models.append(model0)
    plot_two_solutions(data, models[0].solution, models[1].solution, total_customers=total_customers, titles=[f"Phase 1 (params={params[0]})", f"Phase 1 (params={params[1]})"])

def build_phase2_stochastic_matrix(distance_matrix, noise_params=NOISE_PARAMS, seed=None):
    """
    Build one realized Phase-2 (stochastic) matrix.
    We return both the noise_matrix and the realized distance/time matrix for reuse.
    """
    if seed is not None:
        np.random.seed(seed)
    noise_matrix = generate_correlated_mutations(distance_matrix.shape[0], noise_params)
    distance_matrix_stochastic = distance_matrix * noise_matrix
    return noise_matrix, distance_matrix_stochastic

def compute_routing_cost_from_matrix(solution, time_matrix, max_real_node_id):
    """
    Compute routing cost using the realized time_matrix, but handle artificial nodes (XOR).
    We 'compress' each route by removing artificial nodes, and sum costs between consecutive REAL nodes.
    Real nodes are assumed to have ids 0..max_real_node_id.
    """
    total = 0.0
    for r in solution.routes:
        pts = r.point_ids
        # keep only real nodes, in order
        real_pts = [p for p in pts if p <= max_real_node_id]
        # sum consecutive real-real travel times
        for a, b in zip(real_pts[:-1], real_pts[1:]):
            total += float(time_matrix[a][b])
    return total

def count_proxy_trucks_used(solution):
    """
    In your phase2(), proxy truck ids are og_id + 100.
    So vehicle_type_id >= 100 indicates proxy.
    """
    proxy_used = 0
    for r in solution.routes:
        if r.vehicle_type_id >= 100:
            proxy_used += 1
    return proxy_used

def run_two_phase_experiment(path, alpha, beta, total_customers=TOTAL_CUSTOMERS, increase_tw_amount=100, noise_params=NOISE_PARAMS, seed=0, print_solutions=False):
    """
    1) Phase 1 benchmark (α=1,β=0) and Phase 1 buffered (α,beta), plotted side-by-side
    2) Build ONE shared Phase-2 scenario (noise_matrix)
    3) Phase 2 normal (phase2()) vs Phase 2 from scratch (phase1() on realized matrix), plotted side-by-side
    """
    data, distance_matrix = get_data(path, total_customers=total_customers)
    data = increase_time_windows(data, increase_amount=increase_tw_amount)

    # Phase 1 benchmark
    model_p1_bench = phase1(
        data=data,
        distance_matrix=distance_matrix,
        total_customers=total_customers,
        alpha=1.0,
        beta=0.0,
        print_solution=print_solutions,
    )

    # Phase 1 buffered
    model_p1_buf = phase1(
        data=data,
        distance_matrix=distance_matrix,
        total_customers=total_customers,
        alpha=alpha,
        beta=beta,
        print_solution=print_solutions,
    )

    # Plot Phase 1 comparison
    if print_solutions and False:
        plot_two_solutions(
            data,
            model_p1_bench.solution,
            model_p1_buf.solution,
            total_customers=total_customers,

            titles=[
                "Phase 1 benchmark (α=1, β=0)",
                f"Phase 1 buffered (α={alpha}, β={beta})",
            ],
            proxy_extra_cost=0  # not relevant in Phase 1
        )

    # Build ONE shared Phase-2 scenario
    noise_matrix, distance_matrix_stochastic = build_phase2_stochastic_matrix(
        distance_matrix=distance_matrix,
        noise_params=noise_params,
        seed=seed,
    )

    # Phase 2 normal (uses buffered Phase-1 trucks)
    model_p2_normal, proxy_extra_cost_buffer = phase2(
        data=data,
        distance_matrix=distance_matrix,
        model0=model_p1_buf,
        total_customers=total_customers,
        print_solution=print_solutions,
        noise_matrix=noise_matrix,
        noise_params=noise_params,
    )

    # Phase 2 from scratch (does not take into account Phase 1 solution)
    model_p2_scratch = phase1(
        data=data,
        distance_matrix=distance_matrix_stochastic,
        total_customers=total_customers,
        alpha=1.0,
        beta=0.0,
        print_solution=print_solutions,
    )

    # Print raw objectives 
    boris_heeft_gelijk = True

    max_real = total_customers  # depot=0, customers=1..total_customers
    routing_normal = compute_routing_cost_from_matrix(model_p2_normal.solution, distance_matrix_stochastic, max_real_node_id=max_real)
    routing_scratch = compute_routing_cost_from_matrix(model_p2_scratch.solution, distance_matrix_stochastic, max_real_node_id=max_real)

    proxy_count = count_proxy_trucks_used(model_p2_normal.solution)
    total_normal = routing_normal + proxy_count * FIXED_COST_FOR_RELOADING
    total_scratch = routing_scratch  # no switching penalty in scratch
    if print_solutions:
        print("=== FAIR EVALUATION ON SAME REALIZED MATRIX ===")
        print("Routing cost (Phase 2 normal):", routing_normal)
        print("Routing cost (Phase 2 from scratch):", routing_scratch)
        print("Proxy trucks used (normal):", proxy_count)
        print("Switching/reloading cost (normal):", proxy_count * FIXED_COST_FOR_RELOADING)
        print("Total cost (Phase 2 normal):", total_normal)
        print("Total cost (Phase 2 from scratch):", total_scratch)
        plot_solution(data, model_p2_scratch.solution, total_customers=total_customers, custom_cost=routing_scratch)
        plot_two_solutions(
            data,
            model_p1_buf.solution,
            model_p2_normal.solution,
            total_customers=total_customers,
            titles=[
                f"Phase 1, (α={alpha}, β={beta}), cost = {round(model_p1_buf.solution.value)}\n",
                f"Phase 2, (α={alpha}, β={beta}), cost = {round(routing_normal)}\n(w/out switching cost of {proxy_count * FIXED_COST_FOR_RELOADING})",
            ],
            manual_title = True
        )

    return {
        "phase1_benchmark": model_p1_bench.solution.value,
        "phase1_buffered": model_p1_buf.solution.value,
        "phase2_normal_routing": routing_normal,
        "phase2_normal_total": total_normal,
        "phase2_scratch": routing_scratch,
    }

def main_experiment(instance = "c201", 
                alphas=[1], 
                betas = [0], 
                runs_per_instance=10, 
                total_customers = 30, 
                increase_tw_amount=100, 
                noise_params=NOISE_PARAMS):
    setup_logging("main_experiment")
    logging.info(f"Starting main experiment with instance: {instance}, alphas ={alphas}, betas={betas}")
    param_pairs = [(a, b) for a in alphas for b in betas]
    results_total = {param_pair: {
        "phase1_benchmark": [],
        "phase1_buffered": [],
        "phase2_normal_routing": [],
        "phase2_normal_total": [],
        "phase2_scratch": [],
    } for param_pair in param_pairs}

    total_runs = len(param_pairs) * runs_per_instance
    start_time = time.time()
    last_time = start_time
    counter = 0
    for alpha, beta in param_pairs:
        for seed in range(runs_per_instance):
            results = run_two_phase_experiment(
                path=instance,
                alpha=alpha,
                beta=beta,
                total_customers=total_customers,
                increase_tw_amount=increase_tw_amount,
                noise_params=noise_params,
                seed=seed,        
                print_solutions=False,
            )

            for experiment, cost in results.items():
                results_total[(alpha, beta)][experiment].append(cost)
            
            counter += 1
            current = time.time()
            elapsed = current - start_time
            run_time = current - last_time
            avg_time_per_run = elapsed / counter
            est_remaining_time = avg_time_per_run * (total_runs - counter)
            last_time  = current 
            hhmmss = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
            logging.info(f"Completed run {counter}/{total_runs} in {run_time:.2f} sec (Params: {alpha}, {beta}, Seed: {seed}). Estimated remaining time: {hhmmss}.")
    return results_total

def run_single(args):
    # unpack all parameters
    alpha, beta, seed, instance, total_customers, increase_tw_amount, noise_params = args
    results = run_two_phase_experiment(
        path=instance,
        alpha=alpha,
        beta=beta,
        total_customers=total_customers,
        increase_tw_amount=increase_tw_amount,
        noise_params=noise_params,
        seed=seed,
        print_solutions=False,
    )
    return (alpha, beta, seed, results)

def main_experiment_parallel(
    instance="c201",
    alphas=[1],
    betas=[0],
    runs_per_instance=10,
    total_customers=30,
    increase_tw_amount=100,
    noise_params=NOISE_PARAMS,
):
    setup_logging("main_experiment")

    # Create all (alpha, beta, seed, ...) combinations
    param_pairs = [(a, b) for a in alphas for b in betas]
    tasks = [
        (a, b, seed, instance, total_customers, increase_tw_amount, noise_params)
        for a, b in param_pairs
        for seed in range(runs_per_instance)
    ]

    results_total = {
        param_pair: {
            "phase1_benchmark": [],
            "phase1_buffered": [],
            "phase2_normal_routing": [],
            "phase2_normal_total": [],
            "phase2_scratch": [],
        }
        for param_pair in param_pairs
    }

    start_time = time.time()
    counter = 0
    total_runs = len(tasks)

    # Limit workers if needed to avoid overloading laptop
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for alpha, beta, seed, result in executor.map(run_single, tasks):
            for experiment, cost in result.items():
                results_total[(alpha, beta)][experiment].append(cost)
            counter += 1
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / counter
            est_remaining_time = avg_time_per_run * (total_runs - counter)
            hhmmss = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
            logging.info(
                f"Completed run {counter}/{total_runs} "
                f"(Params: {alpha}, {beta}, Seed: {seed}). "
                f"Estimated remaining time: {hhmmss} ."
            )

    return results_total
    
def experiment_summary_to_excel(results_total, filename="results/experiment_results.xlsx"):
    """
    average experiments over the seeds
    the  index should be ethe param_pair
    the columns the experiment types
    and i'm setting the column names manually
    """
    import os
    if not os.path.exists("results"):
        os.makedirs("results")

    base_filename = filename.split(".xlsx")[0].split("/")[-1]
    i = 2
    while os.path.exists(f"results/{filename}.xlsx"):
        filename = f"{base_filename}_{i}"
        i += 1
    summary_data = {}
    for param_pair, experiments in results_total.items():
        summary_data[param_pair] = {}
        for experiment, costs in experiments.items():
            avg_cost = sum(costs) / len(costs) if costs else 0
            summary_data[param_pair][experiment] = avg_cost

    df = pd.DataFrame.from_dict(summary_data, orient='index')

    df = df[[
        "phase1_benchmark",
        "phase1_buffered",
        "phase2_normal_routing",
        "phase2_normal_total",
        "phase2_scratch"
    ]]

    df.to_excel(filename)
    logging.info(f"Experiment summary saved to {filename}")

def setup_logging(filename):
    import os
    i = 2
    og_filename = filename
    while os.path.exists(f"logs/{filename}.log"):
        filename = f"{og_filename}_{i}"
        i += 1

    logging.basicConfig(
        filename=f"logs/{filename}.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

def plot_param_results(results_total, alphas, betas):
    values = {key: results_total[key]['phase2_normal_total'] for key in results_total}
    alpha_results = {alpha: [] for alpha in alphas}
    beta_results = {beta: [] for beta in betas}
    for (alpha, beta), val in values.items():
        print(f"Alpha: {alpha}, Beta: {beta}, Values: {val}")
        print(alpha_results)
        alpha_results[alpha].extend(val)
        beta_results[beta].extend(val)
    alpha_means = {alpha: sum(vals)/len(vals) for alpha, vals in alpha_results.items()}
    alpha_std = {alpha: np.std(vals) for alpha, vals in alpha_results.items()}
    beta_means = {beta: sum(vals)/len(vals) for beta, vals in beta_results.items()}
    beta_std = {beta: np.std(vals) for beta, vals in beta_results.items()}
    #plot alpha and beta side by side, as a line plot, with coloring for std
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].errorbar(list(alpha_means.keys()), list(alpha_means.values()), yerr=list(alpha_std.values()), fmt='-o', capsize=5)
    ax[0].set_title('Average Phase 2 Total Cost vs Alpha')  
    ax[0].set_xlabel('Alpha')
    ax[0].set_ylabel('Average Phase 2 Total Cost')
    ax[1].errorbar(list(beta_means.keys()), list(beta_means.values()), yerr=list(beta_std.values()), fmt='-o', capsize=5)
    ax[1].set_title('Average Phase 2 Total Cost vs Beta')
    ax[1].set_xlabel('Beta')
    ax[1].set_ylabel('Average Phase 2 Total Cost')
    plt.tight_layout()
    plt.show()

def plot_param_results_from_excel(
    excel_path,
    alpha_col_name="alpha",
    beta_col_name="beta",
    value_col="phase2_normal_total"
):
    """
    Reads an experiment summary Excel file and plots
    average Phase 2 total cost vs alpha and beta.
    """

    # Load Excel
    df = pd.read_excel(excel_path, index_col=0)

    # Parse index -> alpha, beta
    if isinstance(df.index[0], str):
        # index like "(alpha, beta)"
        alphas, betas = zip(*[ast.literal_eval(idx) for idx in df.index])
    else:
        # index already tuples
        alphas, betas = zip(*df.index)

    df = df.copy()
    df[alpha_col_name] = alphas
    df[beta_col_name] = betas

    # Group and compute stats
    alpha_stats = df.groupby(alpha_col_name)[value_col].agg(["mean", "std"])
    beta_stats  = df.groupby(beta_col_name)[value_col].agg(["mean", "std"])

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].errorbar(
        alpha_stats.index,
        alpha_stats["mean"],
        yerr=alpha_stats["std"],
        fmt="-o",
        capsize=5
    )
    ax[0].set_title("Average Phase 2 Total Cost vs Alpha")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Average Phase 2 Total Cost")

    ax[1].errorbar(
        beta_stats.index,
        beta_stats["mean"],
        yerr=beta_stats["std"],
        fmt="-o",
        capsize=5
    )
    ax[1].set_title("Average Phase 2 Total Cost vs Beta")
    ax[1].set_xlabel("Beta")
    ax[1].set_ylabel("Average Phase 2 Total Cost")

    plt.tight_layout()
    plt.show()

def compare_param_pairs(p1, p2, N = 50):
    setup_logging("comparison_run")
    results = {p1: [], p2: []}
    start_time = time.time()
    logging.info(f"Starting comparison between params {p1} and {p2} over {N} seeds.")
    for seed in range(N):
        for params in (p1, p2):
            logging.info(f"Running seed {seed+1}/{N} for params {params}.")
            alpha, beta = params
            cost = run_instance(
                path="c201",
                alpha=alpha,
                beta=beta,
                total_customers=40,
                increase_tw_amount=50,
                noise_params={
                    "clip": [0.95, 3],
                    "mean": 1.8,
                    "distr": "gamma"}, 
                seed=seed,
                only_return_cost=True
            )
            results[params].append(cost)
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / ((seed + 1) * 2)
            est_remaining_time = avg_time_per_run * (N * 2 - (seed + 1) * 2)
            hhmmss = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
            logging.info(f"Completed seed {seed+1}/{N} for params {params}. Estimated remaining time: {hhmmss}.")
            logging.info(f"Cost for params {params}, seed {seed}: {cost}")
    logging.info(f"All runs completed. Results: {results}")
    # compute the differences between params per seed
    differences = [results[p1][i] - results[p2][i] for i in range(20)]
    # use wilcoxon signed-rank test to see p2 is lower than p1
    from scipy.stats import wilcoxon
    stat, p_value = wilcoxon(differences, alternative='greater')
    avg_diff = sum(differences) / len(differences)
    logging.info(f"Average cost difference (p1 - p2): {avg_diff}")
    logging.info(f"Wilcoxon signed-rank test statistic: {stat}, p-value: {p_value}")

def plot_results_comparison(results, p1 = (1,0), p2 = (2, 7.5)):
    plt.figure(figsize=(8, 6))
    N = len(results[p1])
    for i in range(N):
        x = results[p1][i]
        y = results[p2][i]
        color = 'green' if y < x else 'red' if x < y else 'gray'
        plt.plot((0,1), (x,y), marker='o', color=color, alpha=0.3)
    plt.xticks([0, 1], [f'Params {p1}', f'Params {p2}'])
    plt.ylabel('Phase 2 Total Cost')
    plt.title('Pairwise omparison of Total Costs')
    plt.tight_layout()
    i = 0
    while os.path.exists(f'plots/comparison_{p1}_vs_{p2}_{i}.png'):
        i += 1
    plt.savefig(f'plots/comparison_{p1}_vs_{p2}_{i}.png')    
    logging.info(f"Comparison plot saved to plots/comparison_{p1}_vs_{p2}_{i}.png")
    return results
    
if __name__ == "__main__":

    #grid search
    alphas = [1.0, 1.5, 2, 2.5, 3]
    betas = [0, 2.5, 5, 7.5, 10, 12.5, 15]
    results_total = main_experiment_parallel(
        instance="c201",
        alphas=alphas,
        betas=betas,
        runs_per_instance=5,
        total_customers=40,
        noise_params={
            "clip": [0.95, 3],
            "mean": 1.8,
            "distr": "gamma"},
        increase_tw_amount=50)
    
    
    plot_param_results(results_total, alphas, betas)
    experiment_summary_to_excel(results_total, filename="results/experiment_results_c201_bigger.xlsx")
    # grid search



    
    # compare_param_pairs(p1 = (1, 0), 
    #                     p2 = (2, 7.5), 
    #                     N=100)


    # actual results hardcoded
    # results = {(1, 0): [1659.9846047030173, 1570.3718624903922, 1634.1680141033237, 2055.528132935422, 1577.444668693822, 1710.9678225213306, 1458.1450445284424, 1526.6640573606257, 1792.8221700485992, 1985.4066256890735, 1447.6448732057165, 1398.6623190534278, 1524.9705850877779, 1547.6248363454063, 1158.8149703158042, 1342.0850992515793, 1403.4966323664855, 1223.6444420306411, 1766.755420570909, 1247.118445849194, 2168.6104182289882, 1412.7697231938685, 1527.1697564841716, 1855.8735198009651, 2010.5896418259217, 1400.76694499658, 1333.8549232173502, 2387.6512664979136, 1454.2767106545994, 1369.0044609508755, 1858.6696566177686, 1151.9409820252945, 2292.613382428679, 1618.0667687245825, 1784.9782883727662, 1716.400434297397, 1885.243304803602, 1454.6601701931293, 1438.5842043058726, 2415.7034517951624, 1655.29983429025, 1606.3389758342341, 2129.3172392241886, 1591.7360519445035, 1634.4415986657073, 1399.8655871895467, 1283.399464477653, 1892.467565041435, 1357.7926870628974, 1549.859270786094, 2051.7142648318277, 2040.5257950657544, 1412.5030182638181, 1810.3884754380858, 1854.6141592109802, 1763.8952727527758, 1191.0250141815031, 1728.304996472647, 1727.3490182380858, 1831.0586859052337, 1637.003464269245, 1059.0742208303404, 1261.2230823677971, 1720.446592367138, 1496.6860190362386, 1337.4807466122065, 2491.88471690998, 1333.597769840813, 2175.8660024623982, 1580.2175042883132, 1848.4994164966909, 1881.9915472182406, 1727.0479683108686, 1704.7354369144628, 2027.695056422889, 1502.550137640575, 1371.7141059289204, 1231.5306663191125, 1795.3288438492175, 2222.250237594888, 1930.847008808746, 1166.1289037483011, 1480.860918443002, 1298.7416358956284, 1056.2884722385206, 1389.5327559966543, 1998.1391363327036, 1643.7216205839559, 1568.9332774147572, 1345.2001400623326, 1604.7153007201011, 1658.6648131673696, 1736.5851913349568, 1644.971053923781, 1355.5723551067124, 1776.0807975241714, 1563.5713774146386, 1438.1010978601855, 1759.5975830039772, 1671.5315507873281], 
    #            (2, 7.5): [1860.638432840484, 1420.158776042318, 1408.5850749501494, 1423.3699648736465, 1373.4242494125015, 1714.3951316838384, 1493.265161714992, 1372.902402102353, 1656.138564624705, 1600.8946208410819, 1619.0048471556518, 1279.9521960991776, 1130.6618676601638, 1120.8517929299894, 1309.2764847377748, 1243.9520111438862, 1456.240609339643, 1087.690677076263, 1968.6401322189017, 1320.1423648302239, 1562.2253186176788, 1542.6023935323433, 1715.2902220570572, 1244.0352833592865, 1193.230764322667, 1358.8546180953501, 1387.62119596581, 2174.6279393085833, 1278.960165731778, 1239.9909397173492, 1689.1922056306432, 835.2108800995527, 1506.4107280813384, 1272.823848642927, 1975.2572785184027, 1483.821692942482, 1477.9198116511157, 1537.4888050788668, 1402.4611089673854, 2149.8834565339816, 1010.6572056073901, 1440.4596888925078, 1493.2817461818827, 1419.2855511479834, 1198.2979159308102, 1333.6791618364773, 1590.2233460336597, 1326.5556017527952, 1231.055430258182, 1337.0108680857707, 1200.376604347608, 0.0, 1295.497423825014, 1758.7352831833264, 1547.9752450091692, 1488.83439822141, 1007.3959225836413, 1672.5407963771302, 1143.4783465069027, 1200.0961930706512, 1311.3007594387734, 973.9620663381351, 1337.04878043161, 1363.403685990817, 1464.0111587404158, 1301.6489312620267, 1841.314917053913, 1248.7036719906712, 1685.320414791475, 1632.9691921765682, 1741.3503975182552, 1134.3628538051375, 1713.0201772123085, 1558.3574449150376, 1412.7994945284536, 1203.6160431742267, 1488.1205809901428, 1245.42051789405, 1382.305512348686, 1680.6964935975682, 1718.7503531265388, 1144.6564643052402, 1290.9704986679737, 1316.958698801015, 1286.9637467023745, 1297.6381690056826, 1396.4383174707891, 1254.0674628025629, 1832.4990874013868, 1166.0783020037366, 1232.317111862411, 1333.1486569188548, 1468.106511138111, 1796.5580493893547, 1373.1686664442946, 1333.9828071263005, 1406.2780101570065, 1138.6278775073865, 1564.729588343806, 1439.188446883954]}
    # plot_results_comparison(results)

