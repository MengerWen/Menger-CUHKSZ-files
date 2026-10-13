import cvxpy as cp
import numpy as np

# Transportation cost matrix (direct costs)
cost_matrix = np.array([
    [0, 5, 3, 9, 6],   # A
    [5, 0, 9, 7, 4],   # B
    [3, 9, 0, 4, 6],   # C
    [9, 7, 4, 0, 8],   # 1
    [6, 4, 6, 8, 0]    # 2
])

# Floyd-Warshall Algorithm to find the shortest path (indirect costs)
n = len(cost_matrix)
# Using the Floyd-Warshall algorithm to calculate shortest paths
next_site = np.array([[j if i != j else -1 for j in range(n)] for i in range(n)])  # For backtracking the path

# Run the Floyd-Warshall Algorithm to compute shortest paths and backtrack information
for k in range(n):
    for i in range(n):
        for j in range(n):
            if cost_matrix[i, j] > cost_matrix[i, k] + cost_matrix[k, j]:
                cost_matrix[i, j] = cost_matrix[i, k] + cost_matrix[k, j]
                next_site[i, j] = next_site[i, k]

# Decision variables (integer variables)
x_A1 = cp.Variable(integer=True)
x_A2 = cp.Variable(integer=True)
x_B1 = cp.Variable(integer=True)
x_B2 = cp.Variable(integer=True)
x_C1 = cp.Variable(integer=True)
x_C2 = cp.Variable(integer=True)

# Objective function (total cost), using the updated cost matrix
cost = 0.1 * (
    cost_matrix[0, 3] * x_A1 + cost_matrix[0, 4] * x_A2 +  # A -> 1, A -> 2
    cost_matrix[1, 3] * x_B1 + cost_matrix[1, 4] * x_B2 +  # B -> 1, B -> 2
    cost_matrix[2, 3] * x_C1 + cost_matrix[2, 4] * x_C2    # C -> 1, C -> 2
)

# Constraints
constraints = [
    x_A1 + x_A2 <= 7000,  # 7000 rods at A
    x_B1 + x_B2 <= 4000,  # 4000 rods at B
    x_C1 + x_C2 <= 3500,  # 3500 rods at C
    x_A1 + x_B1 + x_C1 >= 8000,  # 8000 rods needed at site 1
    x_A2 + x_B2 + x_C2 >= 5500,  # 5500 rods needed at site 2
    x_A1 >= 0, x_A2 >= 0, x_B1 >= 0, x_B2 >= 0, x_C1 >= 0, x_C2 >= 0  # Non-negativity
]

# Problem definition
problem = cp.Problem(cp.Minimize(cost), constraints)

# Solve the problem
problem.solve()

# Results (modified format)
print("\nOptimal allocation of rods:")
sites = ["A", "B", "C"]
values = [x_A1.value, x_A2.value, x_B1.value, x_B2.value, x_C1.value, x_C2.value]

# Output total cost
print("\nTotal optimal cost:", problem.value)
print(f"x_A1: {x_A1.value}, x_A2: {x_A2.value}")
print(f"x_B1: {x_B1.value}, x_B2: {x_B2.value}")
print(f"x_C1: {x_C1.value}, x_C2: {x_C2.value}\n")

# Function to backtrack and print the path
def get_path(start, end):
    path = []
    while start != end:
        path.append(start)
        start = next_site[start, end]
    path.append(end)
    return path

# Print the rod movements in the specified format, including transit if any
for i in range(3):
    for j in range(2):
        quantity = values[i*2 + j]
        if quantity > 0:
            # Get the path from site i to site j
            path = get_path(i, j + 3)  # 1 and 2 are indexed at 3 and 4
            path_str = " -> ".join([sites[p] if p < 3 else str(p-2) for p in path])
            print(f"Move {quantity} rods from site {sites[i]} to site {j+1} via {path_str}")