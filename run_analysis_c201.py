# runs parameter comparison experiments for C201

from cvrptw1 import run_two_phase_experiment
from constants import NOISE_PARAMS

INSTANCE = "c205"
TOTAL_CUSTOMERS = 50

PARAM_SETS = [
    (1.00, 0),
    (1.05, 5),
    (1.15, 10),
    (1.30, 15),
    (1.40, 20),
]

def main():
    for alpha, beta in PARAM_SETS:
        print("\n==============================")
        print(f"Running {INSTANCE} with alpha={alpha}, beta={beta}")

        run_two_phase_experiment(
            path=INSTANCE,
            alpha=alpha,
            beta=beta,
            total_customers=TOTAL_CUSTOMERS,
            increase_tw_amount=100,
            noise_params=NOISE_PARAMS,
            seed=0,                 # same scenario every time (fair comparison)
            print_solutions=False,
        )

if __name__ == "__main__":
    main()
