from cvrptw1 import get_data, phase1, phase2
from constants import *
from distance_mutator import generate_correlated_mutations
from functools import partial
#TODO: build safety for if alpha is too high and phase 1 already crashes

from contextlib import redirect_stdout, redirect_stderr
import io

def run_instances(data, distance_matrix,
                  alphas, 
                  total_customers=TOTAL_CUSTOMERS,  
                  noise_params=NOISE_PARAMS):
    #path can either be of form 'c201' or 'In/c201.txt'
    noise_matrix = generate_correlated_mutations(len(distance_matrix), noise_params=noise_params)
    result = []
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):

        for alpha in alphas:
            
            
            model0 = phase1(data, distance_matrix, 
                            alpha=alpha, 
                            total_customers=total_customers, 
                            print_solution=False)
            
            model1 = phase2(data, distance_matrix, 
                            model0, 
                            total_customers=total_customers, 
                            print_solution=False, 
                            noise_matrix=noise_matrix)
            
            if model1.solution.is_defined():
                result.append(model1.solution.value)
    return tuple(result)


import numpy as np

def SPSA(
    eval_function,
    alpha_init=1.0,
    a=0.15,          # constant learning rate
    c=0.02,          # constant perturbation size
    alpha_range=(0.1, 2),
    max_iter=100
):
    """
    1D SPSA with:
    - constant step size
    - common random numbers (CRN)
    - single function call per iteration    
    """

    alpha = alpha_init
    alpha_trace = [alpha]
    for _ in range(max_iter):
        delta = np.random.choice([-1.0, 1.0])

        alpha_plus = np.clip(alpha + c * delta, *alpha_range)
        alpha_minus = np.clip(alpha - c * delta, *alpha_range)

        # CRN: single call evaluates both
        y_plus, y_minus = eval_function(alphas=(alpha_plus, alpha_minus))

        # SPSA gradient estimate
        g_hat = (y_plus - y_minus) / (2.0 * c * delta)
        # Gradient step
        alpha = alpha - a * g_hat
        alpha = np.clip(alpha, *alpha_range)
        alpha_trace.append(alpha)


    return alpha_trace

def plot_trace(trace):
    import matplotlib.pyplot as plt

    plt.plot(trace, marker='o')
    plt.title('SPSA Optimization Trace')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha Value')
    plt.grid()
    plt.show()

def main():
    #change this to be dynamic
    total_customers = TOTAL_CUSTOMERS
    noise_params = NOISE_PARAMS
    path = 'c201'
    data, distance_matrix = get_data(path, total_customers=total_customers)
    eval_function = partial(run_instances, data=data, distance_matrix=distance_matrix, total_customers=total_customers, noise_params=noise_params)
    alpha_trace = SPSA(eval_function, alpha_init=1.0, a=0.05, c=0.02, alpha_range=(0.9, 1.5), max_iter=50)
    plot_trace(alpha_trace)
main()