from unittest import result
from cvrptw1 import get_data, phase1, phase2, increase_time_windows
from constants import *
from distance_mutator import generate_correlated_mutations
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

#TODO: build safety for if phases crash more often


def run_instances(
    data,
    distance_matrix,
    param_sets=None,
    total_customers=TOTAL_CUSTOMERS,
    noise_params=NOISE_PARAMS,
    penalty=1e9,
):
    noise_matrix = generate_correlated_mutations(len(distance_matrix), noise_params=noise_params)

    def eval_one(params):
        alpha, beta = float(params[0]), float(params[1])

        try:
            model0 = phase1(
                data, distance_matrix,
                alpha=alpha, beta=beta,
                total_customers=total_customers,
                print_solution=False
            )
        except Exception:
            return penalty

        try:
            model1, proxy_extra_cost = phase2(
                data, distance_matrix, model0,
                total_customers=total_customers,
                print_solution=False,
                noise_matrix=noise_matrix
            )
        except Exception:
            return penalty

        if not model1.solution.is_defined():
            return penalty

        return model1.solution.value - proxy_extra_cost

    # IMPORTANT: always return one value per param set
    return tuple(eval_one(p) for p in param_sets)


import numpy as np
import matplotlib.pyplot as plt

def SPSA(
    eval_function,
    params_init=(1.0, 5.0),   # (alpha, beta)
    a=np.array([0.01, 0.25]),      # constant learning rate
    c=np.array([0.05, 3]),         # constant perturbation size
    alpha_range=(0.1, 2.0),
    beta_range=(-5.0, 20.0),
    max_iter=100,
    logger=None
):
    """
    2D SPSA with:
    - constant step size
    - common random numbers (CRN)
    - single function call per iteration
    """

    params = np.array(params_init, dtype=float)
    trace = [params.copy()]

    for iteration in range(max_iter):
        alpha, beta = params

        if logger is not None:
            logger.info(
                f"SPSA Iteration {iteration+1}/{max_iter}, "
                f"alpha={alpha:.4f}, beta={beta:.4f}"
            )

        # 2D Rademacher perturbation
        delta = np.random.choice([-1.0, 1.0], size=2)

        params_plus = params + c * delta
        params_minus = params - c * delta

        # Clip per dimension
        params_plus[0] = np.clip(params_plus[0], *alpha_range)
        params_minus[0] = np.clip(params_minus[0], *alpha_range)
        params_plus[1] = np.clip(params_plus[1], *beta_range)
        params_minus[1] = np.clip(params_minus[1], *beta_range)

        # CRN: single call returns both evaluations
        y_plus, y_minus = eval_function(param_sets = (params_plus, params_minus))
        
        if not np.isfinite(y_plus) or not np.isfinite(y_minus):
            raise ValueError(f"Non-finite objective returned: y_plus={y_plus}, y_minus={y_minus}")


        if logger is not None:
            logger.info(
                f"  Evaluated at:\n"
                f"  + alpha={params_plus[0]:.4f}, beta={params_plus[1]:.4f}, cost={y_plus:.2f}\n"
                f"  - alpha={params_minus[0]:.4f}, beta={params_minus[1]:.4f}, cost={y_minus:.2f}"
            )

        # SPSA gradient estimate (vector)
        g_hat = (y_plus - y_minus) / (2.0 * c * delta)

        # Gradient step
        params = params - a * g_hat

        # Clip again after update
        params[0] = np.clip(params[0], *alpha_range)
        params[1] = np.clip(params[1], *beta_range)

        trace.append(params.copy())

    return np.array(trace)


def plot_trace(trace):
    trace = np.asarray(trace)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trace[:, 0], marker='o')
    plt.title('Alpha trace')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(trace[:, 1], marker='o')
    plt.title('Beta trace')
    plt.xlabel('Iteration')
    plt.ylabel('Beta')
    plt.grid()

    plt.tight_layout()
    plt.show()


def main():
    logger = logging.getLogger(__name__)
    #change this to be dynamic
    total_customers = 30
    noise_params = NOISE_PARAMS
    path = 'c101'
    logger.info(f"Starting SPSA optimization for VRPTW with {total_customers}.")
    
    data, distance_matrix = get_data(path, total_customers=total_customers)
    data = increase_time_windows(data, increase_amount=100)

    eval_function = partial(run_instances, data=data, distance_matrix=distance_matrix, total_customers=total_customers, noise_params=noise_params)
    alpha_trace = SPSA(eval_function, 
                       params_init=(1.5, 10), 
                       a=np.array([0.05, 0.25]), 
                       c=np.array([0.05, 2]),  
                       alpha_range=(1, 2),
                       beta_range = (0, 20), 
                       max_iter=15, 
                       logger = logger)
    plot_trace(alpha_trace)

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="logs/vrp_experiment.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    main()