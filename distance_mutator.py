import numpy as np
import matplotlib.pyplot as plt

example_array_7x7 = np.array([
    [0, 1, 2, 3, 4, 5, 6],
    [1, 0, 1, 2, 3, 4, 5],
    [2, 1, 0, 1, 2, 3, 4],
    [3, 2, 1, 0, 1, 2, 3],
    [4, 3, 2, 1, 0, 1, 2],
    [5, 4, 3, 2, 1, 0, 1],
    [6, 5, 4, 3, 2, 1, 0]
])


def generate_randoms(n, clip = [0.9, 3], mean = 1, distr = "gamma"):
    """Generate n random numbers, exponentially distributed."""
    if distr == "gamma":
        np.random.gamma(2, scale=1, size=n)
    else:
        print("<generate_randoms> Warning: Unknown distribution, defaulting to exponential.")
        randoms = np.random.exponential(mean, n)
    if clip: randoms = np.clip(randoms, clip[0], clip[1])
    return randoms

def generate_correlated_mutations(n, clip = [0.9, 2], mean = 1.4):
    """Generate n correlated random numbers, exponentially distributed."""
    row_factors = generate_randoms(n, clip, mean)
    col_factors = generate_randoms(n, clip, mean).reshape((n, 1))
    mutations = row_factors * col_factors
    print(mutations)
    return mutations

def colorplot_matrix(matrix, title="Matrix", xlabel="X", ylabel="Y"):
    """Create a color plot of a matrix."""
    plt.imshow(matrix, cmap='Reds', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# mutations = generate_correlated_mutations(10)
# colorplot_matrix(mutations, title="Distance Mutation Matrix", xlabel="Points", ylabel="Points")
    

