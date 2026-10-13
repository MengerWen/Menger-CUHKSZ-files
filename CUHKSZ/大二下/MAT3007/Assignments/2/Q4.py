import pulp
import matplotlib.pyplot as plt
import numpy as np

# Define the LP problem
prob = pulp.LpProblem("Largest_Square", pulp.LpMaximize)

# Variables
x = pulp.LpVariable('x', lowBound=0)
y = pulp.LpVariable('y', lowBound=0)
s = pulp.LpVariable('s', lowBound=0)

# Objective function
prob += s

# Constraints from the edges
prob += 4*x + y >= 8
prob += 3*x + 3*s - 8*y <= 6
prob += 7*x + 7*s - 2*y <= 64
prob += 2*x + 5*y + 7*s <= 74
prob += y + s <= 12
prob += 4*x - 3*y >= -24 + 3*s

# Solve the problem
prob.solve()

# Extract results
x_val = x.value()
y_val = y.value()
s_val = s.value()
area_val = s_val**2

# Output optimal solution in a more formal way
print(f"Optimal Solution for Largest Square Inscribed in Polyhedron P:")
print(f"  - Bottom-left corner coordinates: x = {x_val:.6f}, y = {y_val:.6f}")
print(f"  - Side length of the square: s = {s_val:.6f}")
print(f"  - Area of the square: {area_val:.6f}")

# Define polygon vertices
vertices = np.array([
    [0, 8],
    [2, 0],
    [10, 3],
    [12, 10],
    [7, 12],
    [3, 12]
])

# Square coordinates
square = np.array([
    [x_val, y_val],
    [x_val + s_val, y_val],
    [x_val + s_val, y_val + s_val],
    [x_val, y_val + s_val],
    [x_val, y_val]
])

# Plotting
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Plot polygon
ax.fill(vertices[:,0], vertices[:,1], 'lightgray', edgecolor='black', label='Polyhedron P')

# Plot square
ax.fill(square[:,0], square[:,1], 'blue', alpha=0.5, label='Largest Square')

# Annotate points
ax.plot(vertices[:,0], vertices[:,1], 'o', color='black')
ax.plot(square[:,0], square[:,1], 'o', color='blue')

ax.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Largest Axis-Aligned Square Inscribed in Polyhedron P')
plt.grid(True)
plt.show()