import numpy as np

data = np.loadtxt("/home/hassene/Desktop/Stage_code/mult_dimensional_grids/1450_2_nopti")

# Exclude last row (global distortions)
points = data[:-1, 1:3]    # Shape (1450, 2)
weights = data[:-1, 0]     # Shape (1450,)
local_L2 = data[:-1, 3]    # Shape (1450,)
local_L1 = data[:-1, 4]    # Shape (1450,)

# Global distortions
global_L2 = data[-1, 3]
global_L1 = data[-1, 4]
def gaussian_expectation(f, mu, Sigma, points=points, weights=weights):
    A = np.linalg.cholesky(Sigma)  # Cholesky: Sigma = A @ A.T

    # Transform quantization points from N(0, I_2) to N(mu, Sigma)
    transformed_points = points @ A.T + mu  # Shape (N, 2)

    # Evaluate f at each transformed point
    f_vals = np.array([f(x, y) for x, y in transformed_points])

    # Weighted sum
    return np.sum(f_vals * weights)


data_1 = np.loadtxt("/home/hassene/Desktop/Stage_code/one_dim_1001_5999/5999_1_nopti")


# Exclude last row (global distortions)
points = data[:-1, 1:3]    # Shape (1450, 2)
weights = data[:-1, 0]     # Shape (1450,)
local_L2 = data[:-1, 3]    # Shape (1450,)
local_L1 = data[:-1, 4]    # Shape (1450,)

# Global distortions
global_L2 = data[-1, 3]
global_L1 = data[-1, 4]
def gaussian_expectation(f, mu, Sigma, points=points, weights=weights):
    A = np.sqrt(Sigma)  # Cholesky: Sigma = A @ A.T

    # Transform quantization points from N(0, I_2) to N(mu, Sigma)
    transformed_points = points * A + mu  # Shape (N, 2)
    
    # Evaluate f at each transformed point
    f_vals = np.array([f(x) for x in transformed_points])

    # Weighted sum
    return np.sum(f_vals * weights)


data_1 = np.loadtxt("/home/hassene/Desktop/Stage_code/one_dim_1001_5999/5999_1_nopti")

# Quantization points and weights for 1D
points = data_1[:-1, 1]     # shape (N,)
weights = data_1[:-1, 0]    # shape (N,)

def gaussian_expectation_1(f, mu, Sigma, points=points, weights=weights):
    A = np.sqrt(Sigma)  # Sigma is scalar in 1D

    transformed_points = points * A + mu  # shape (N,)
    f_vals = np.array([f(x) for x in transformed_points])  # shape (N,)

    return np.sum(f_vals * weights)

