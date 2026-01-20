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

def compute_euclidean_distance(x_i, x_j, y_i, y_j):
  """compute the euclidean distance between 2 points from graph"""
  return round(math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2), 3)
 
def plot_solution(data, solution):
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
                if 0 <= cust_idx < len(data['cust_coordinates']):
                    x_coords.append(data['cust_coordinates'][cust_idx][0])
                    y_coords.append(data['cust_coordinates'][cust_idx][1])
        
        # Kies een kleur
        color = colors[i % len(colors)]
        
        # Teken de lijn
        plt.plot(x_coords, y_coords, c=color, linewidth=2, label=f'Route {i+1}', alpha=0.8)
        
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

def plot_two_solutions(data, solution1, solution2, titles=None):
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
                    if 0 <= cust_idx < len(data['cust_coordinates']):
                        x_coords.append(data['cust_coordinates'][cust_idx][0])
                        y_coords.append(data['cust_coordinates'][cust_idx][1])
            color = colors[i % len(colors)]
            ax.plot(x_coords, y_coords, c=color, linewidth=2, label=f'Route {i+1}', alpha=0.8)
            
            # Optionele pijltjes
            for k in range(len(x_coords) - 1):
                mid_x = (x_coords[k] + x_coords[k+1]) / 2
                mid_y = (y_coords[k] + y_coords[k+1]) / 2
                dx = (x_coords[k+1] - x_coords[k]) * 0.1
                dy = (y_coords[k+1] - y_coords[k]) * 0.1
                ax.arrow(mid_x, mid_y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=1.5, color=color)

        ax.set_title(f"{title} (Kosten: {round(sol.value, 2)})")
        ax.set_xlabel('X Coördinaat')
        ax.set_ylabel('Y Coördinaat')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    
    plt.tight_layout()
    plt.show()


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


def phase1(data, distance_matrix, total_customers=TOTAL_CUSTOMERS, alpha = 1, print_solution=True):
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
            dist = distance_matrix[i][j] *alpha
            model0.add_link(
                start_point_id=i, 
                end_point_id=j, 
                distance=dist, 
                time=dist
            )

    # model solven
    model0.set_parameters(time_limit=100, solver_name="CLP")
    model0.solve()
    if print_solution:
        print(model0.solution)
        plot_solution(data, model0.solution)
    

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

    # verschillende trucks per load, max 1
    for i, capacity in enumerate(load_per_truck):
        model1.add_vehicle_type(
            id=i + 1,
            start_point_id=0,
            end_point_id=0,
            max_number=1,
            capacity=capacity,
            tw_begin=data["depot_tw_begin"],
            tw_end=data["depot_tw_end"],
            var_cost_time=1, 
            # var_cost_dist=1
        )

    # Twee keer zo veel trucks met max capacity  
    # Het max aantal staat hier weer op het originele max aantal, 
    # dus je kan twee keer te veel trucks hebben.
    # ook een fixed cost toegevoegd, zodat de originele trucks zo veel mogelijk worden gebruikt.
    model1.add_vehicle_type(id=number_of_trucks + 1,
        start_point_id=0,
        end_point_id=0,
        max_number=data["max_number"],
        capacity=data["vehicle_capacity"],
        tw_begin=data["depot_tw_begin"],
        tw_end=data["depot_tw_end"],
        var_cost_time=1,
        # var_cost_dist=1,
        fixed_cost=FIXED_COST_PER_TRUCK
        )


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

    # model solven en plotten
    model1.set_parameters(time_limit=100, solver_name="CLP")
    model1.solve()
    if print_solution:
        print(model1.solution)
        plot_solution(data, model1.solution)
    return model1


def run_instance(path,
                  alpha, 
                  total_customers=TOTAL_CUSTOMERS,  
                  noise_params=NOISE_PARAMS, 
                  print_solution=False,
                  plot_both_solutions=False):
    #path can either be of form 'c201' or 'In/c201.txt'
    data, distance_matrix = get_data(path, total_customers=total_customers)

    model0 = phase1(data, distance_matrix, 
                    alpha=alpha, 
                    total_customers=total_customers, 
                    print_solution=print_solution)

    model1 = phase2(data, distance_matrix, 
                    model0, 
                    total_customers=total_customers, 
                    print_solution=print_solution, 
                    noise_params=noise_params)
    
    if plot_both_solutions:
        plot_two_solutions(data, model0.solution, model1.solution, titles=["Phase 1", "Phase 2"])
    return model0, model1

if __name__ == "__main__":
    #ex 
    model0, model1 = run_instance('c205', alpha=1, total_customers=50, plot_both_solutions=True, noise_params=NOISE_PARAMS)

















