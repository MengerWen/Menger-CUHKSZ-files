import cvxpy as cp

# Define the rooms and the adjacency relationships based on the given floor plan
rooms = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
adjacency = {
    'A': ['B', 'D'],
    'B': ['A', 'C', 'E', 'F'],
    'C': ['B', 'G'],
    'D': ['A', 'E', 'H'],
    'E': ['B', 'H'],
    'F': ['B', 'G'],
    'G': ['C', 'F', 'J'],
    'H': ['D', 'E', 'I'],
    'I': ['H'],
    'J': ['G']
}

# Define the binary decision variable for guards at the doors between rooms
x = {}
for i in rooms:
    for j in adjacency[i]:
        # Ensure each door is represented in only one direction (e.g., A-B or B-A)
        if i < j:
            x[i, j] = cp.Variable(boolean=True)

# Objective function: Minimize the total number of guards
objective = cp.Minimize(sum(x[i, j] for i, j in x))

# Constraints: Each room must be covered by at least one guard
constraints = []
for room in rooms:
    constraints.append(
        sum(x[min(room, adj), max(room, adj)] for adj in adjacency[room]) >= 1
    )

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Retrieve the results and display
solution = {}
for i, j in x:
    solution[(i, j)] = x[i, j].value

print("Solution:")
for pair, value in solution.items():
    if value > 0.5:  # Display only the active guard assignments
        print(f"Guard placed at door between rooms {pair[0]} and {pair[1]}")