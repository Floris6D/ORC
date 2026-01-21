#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:46:01 2026

@author: martijnkrikke
"""

import matplotlib.pyplot as plt
import VRPSolverEasy as solver
import math
import numpy as np
from scipy.spatial.distance import cdist
from distance_mutator import generate_correlated_mutations
from constants import *



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
 
def plot_solution(data, solution, total_customers):
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
    plt.title(f"Oplossing C201 (Kosten: {round(solution.value, 2)})")
    plt.xlabel('X Coördinaat')
    plt.ylabel('Y Coördinaat')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_two_solutions(data, solution1, solution2, total_customers, titles=None, proxy_extra_cost=0):
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
        if "phase 2" in title.lower() or "phase2" in title.lower():
            ax.set_title(f"{title} (Kosten: {round(sol.value - proxy_extra_cost, 2)} excl. proxy kosten)")
        ax.set_xlabel('X Coördinaat')
        ax.set_ylabel('Y Coördinaat')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    
    plt.tight_layout()
    plt.show()


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
        # var_cost_dist = 1
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
        plot_solution(data, model0.solution, total_customers=total_customers)
    

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
            # var_cost_dist=1,
            fixed_cost=VERY_HIGH_COST,
            name=f"OG truck {i+1}"
        )

        #Proxy trucks
        proxy_truck_id  = og_truck_id + 100
        model1.add_vehicle_type(id=proxy_truck_id,
            start_point_id=0,
            end_point_id=0,
            max_number=1,
            capacity=data["vehicle_capacity"]+1, #+1 for going through the XOR node with demand=1
            tw_begin=data["depot_tw_begin"],
            tw_end=data["depot_tw_end"],
            var_cost_time=1,
            # var_cost_dist=1,
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
        plot_solution(data, model1.solution, total_customers=total_customers)
    return model1, proxy_extra_cost

def run_instance(path,
                  alpha, 
                  beta,
                  total_customers=TOTAL_CUSTOMERS,  
                  noise_params=NOISE_PARAMS, 
                  print_solution=False,
                  plot_both_solutions=False,
                  increase_tw_amount=0):
    #path can either be of form 'c201' or 'In/c201.txt'
    data, distance_matrix = get_data(path, total_customers=total_customers)
    data = increase_time_windows(data, increase_amount=100)

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
                    noise_params=noise_params)
    
    if plot_both_solutions:
        plot_two_solutions(data, model0.solution, model1.solution, total_customers=total_customers,  titles=["Phase 1", "Phase 2"], proxy_extra_cost=proxy_extra_cost)
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
    model_p2_normal, proxy_extra_cost = phase2(
        data=data,
        distance_matrix=distance_matrix,
        model0=model_p1_buf,
        total_customers=total_customers,
        print_solution=print_solutions,
        noise_matrix=noise_matrix,
        noise_params=noise_params,
    )

    # Phase 2 from scratch (does not take into account Phase 1 solution)
    # IMPORTANT: alpha=1, beta=0 so we do not apply buffering again
    model_p2_scratch = phase1(
        data=data,
        distance_matrix=distance_matrix_stochastic,
        total_customers=total_customers,
        alpha=1.0,
        beta=0.0,
        print_solution=print_solutions,
    )

    # Print raw objectives
    max_real = total_customers  # depot=0, customers=1..total_customers
    routing_normal = compute_routing_cost_from_matrix(model_p2_normal.solution, distance_matrix_stochastic, max_real_node_id=max_real)
    routing_scratch = compute_routing_cost_from_matrix(model_p2_scratch.solution, distance_matrix_stochastic, max_real_node_id=max_real)

    proxy_count = count_proxy_trucks_used(model_p2_normal.solution)
    total_normal = routing_normal + proxy_count * FIXED_COST_FOR_RELOADING
    total_scratch = routing_scratch  # no switching penalty in scratch

    print("=== FAIR EVALUATION ON SAME REALIZED MATRIX ===")
    print("Routing cost (Phase 2 normal):", routing_normal)
    print("Routing cost (Phase 2 from scratch):", routing_scratch)
    print("Proxy trucks used (normal):", proxy_count)
    print("Switching/reloading cost (normal):", proxy_count * FIXED_COST_FOR_RELOADING)
    print("Total cost (Phase 2 normal):", total_normal)
    print("Total cost (Phase 2 from scratch):", total_scratch)


    plot_two_solutions(
        data,
        model_p2_scratch.solution,
        model_p2_normal.solution,
        total_customers=total_customers,
        titles=[
            "Phase-2 from scratch",
            f"Phase 2 normal (α={alpha}, β={beta})"
        ],
        proxy_extra_cost=proxy_extra_cost
    )


if __name__ == "__main__":
    #ex 
    INSTANCE = "c201"         
    ALPHA = 1.3
    BETA = 5                     

    run_two_phase_experiment(
        path=INSTANCE,
        alpha=ALPHA,
        beta=BETA,
        total_customers=30,
        increase_tw_amount=100,
        noise_params=NOISE_PARAMS,
        seed=0,        
        print_solutions=False,
    )
    '''
    model0, model1 = run_instance('c205', 
                                  alpha=1.05, 
                                  beta=5, 
                                  total_customers=30, 
                                  plot_both_solutions=True,
                                  print_solution=True, 
                                  noise_params=NOISE_PARAMS,
                                  increase_tw_amount=100
                                  )
    '''
    # _test_run_multiple_params_phase1('rc107', params=[(1.0, 0), (1.5, 10)], total_customers=100)
