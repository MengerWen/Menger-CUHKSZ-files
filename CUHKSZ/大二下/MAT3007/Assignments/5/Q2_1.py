# Import the required library
from scipy.optimize import linprog

# Define the coefficients of the objective function (negated for maximization)
c = [-5, -10]  # We minimize -5x_1 - 10x_2 to maximize 5x_1 + 10x_2

# Define the coefficient matrix for the inequality constraints (A_ub x <= b_ub)
A = [
    [1, 3],  # x_1 + 3x_2 <= 50
    [4, 2],  # 4x_1 + 2x_2 <= 60
    [1, 0]   # x_1 <= 5
]

# Define the right-hand side of the inequality constraints
b = [50, 60, 5]

# Define bounds for x_1 and x_2 (both >= 0)
bounds = [(0, None), (0, None)]

# Solve the linear programming problem
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# Extract the optimal solution
optimal_solution = result.x

# Calculate the optimal value (negate the result since we minimized the negative)
optimal_value = -result.fun

# Print the results
print("Optimal solution: x1 =", optimal_solution[0], ", x2 =", optimal_solution[1])
print("Optimal value:", optimal_value)