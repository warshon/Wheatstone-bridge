# Import required libraries
from skopt import gp_minimize
from skopt.space import Real

a = 1.0
b = 100.0
# Define the Rosenbrock function (2D case)
def rosenbrock(x):
    """
    Computes the Rosenbrock function for a 2D input vector.
    
    Parameters:
    x (list): Input vector [x1, x2]
    
    Returns:
    float: Function value at x
    """
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

# Define the search space for variables
dimensions = [
    Real(0.0, 2.0, name='x1', prior='uniform'),  # x1 ∈ [0, 2]
    Real(0.0, 2.0, name='x2', prior='uniform')   # x2 ∈ [0, 2]
]

# Configure and execute Bayesian optimization
result = gp_minimize(
    func=rosenbrock,           # Objective function
    dimensions=dimensions,     # Variable search space
    n_calls=150,               # Total function evaluations
    n_random_starts=10,        # Initial random evaluations
    random_state=42,           # Seed for reproducibility
    noise=1e-10,               # Assume noiseless observations
    verbose=True               # Display progress
)

# Extract and print optimization results
optimal_params = result.x
min_value = result.fun
print("\nOptimization Results:")
print(f"Global minimum at: x1 = {optimal_params[0]:.6f}, x2 = {optimal_params[1]:.6f}")
print(f"Minimum function value: {min_value:.6f}")
print(f"Number of function evaluations: {len(result.func_vals)}")