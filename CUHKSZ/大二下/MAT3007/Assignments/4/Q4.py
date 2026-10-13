from pulp import *

# Solve the Primal Problem

# Define the primal problem
primal = LpProblem("Transportation_Primal", LpMinimize)

# Define decision variables with non-negativity
x11 = LpVariable("x11", lowBound=0)
x12 = LpVariable("x12", lowBound=0)
x13 = LpVariable("x13", lowBound=0)
x21 = LpVariable("x21", lowBound=0)
x22 = LpVariable("x22", lowBound=0)
x23 = LpVariable("x23", lowBound=0)

# Set the objective function
primal += 6*x11 + 8*x12 + 10*x13 + 7*x21 + 5*x22 + 9*x23, "Total_Cost"

# Add supply constraints
primal += x11 + x12 + x13 <= 500, "W1_Supply"
primal += x21 + x22 + x23 <= 400, "W2_Supply"

# Add demand constraints
primal += x11 + x21 >= 300, "S1_Demand"
primal += x12 + x22 >= 400, "S2_Demand"
primal += x13 + x23 >= 200, "S3_Demand"

# Solve the primal problem
primal.solve()

# Print the results
print("Primal Problem Results:")
print("Status:", LpStatus[primal.status])
print("Optimal Solution:")
for v in primal.variables():
    print(f"{v.name} = {v.varValue}")
print("Total Cost (Primal Objective) =", value(primal.objective))

# Solve the Dual Problem

# Define the dual problem
dual = LpProblem("Transportation_Dual", LpMaximize)

# Define dual variables with appropriate bounds
u1 = LpVariable("u1", upBound=0)  # u1 <= 0
u2 = LpVariable("u2", upBound=0)  # u2 <= 0
v1 = LpVariable("v1", lowBound=0)  # v1 >= 0
v2 = LpVariable("v2", lowBound=0)  # v2 >= 0
v3 = LpVariable("v3", lowBound=0)  # v3 >= 0

# Set the objective function
dual += 500*u1 + 400*u2 + 300*v1 + 400*v2 + 200*v3, "Dual_Objective"

# Add constraints
dual += u1 + v1 <= 6, "Constraint_11"
dual += u1 + v2 <= 8, "Constraint_12"
dual += u1 + v3 <= 10, "Constraint_13"
dual += u2 + v1 <= 7, "Constraint_21"
dual += u2 + v2 <= 5, "Constraint_22"
dual += u2 + v3 <= 9, "Constraint_23"

# Solve the dual problem
dual.solve()

# Print the results
print("\nDual Problem Results:")
print("Status:", LpStatus[dual.status])
print("Optimal Solution:")
for v in dual.variables():
    print(f"{v.name} = {v.varValue}")
print("Dual Objective =", value(dual.objective))