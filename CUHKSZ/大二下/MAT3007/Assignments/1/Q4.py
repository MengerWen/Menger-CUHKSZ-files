import cvxpy as cp

# Corrected item values (individual diamonds, combined dogs)
values = [
    8.0,   # Sketch
    0.5,   # Bust
    3.5,   # Vase
    6.0,   # Porsche
    1.2,   # Diamond 1
    1.2,   # Diamond 2
    1.2,   # Diamond 3
    0.3,   # Sofa
    0.6,   # Dogs (combined)
    1.0,   # Sculpture
    2.0,   # Boat
    1.0    # Motorbike
]
n_items = len(values)  # Now 12 items
n_daughters = 3

# Binary variables: x[i,j] =1 if item i goes to daughter j
x = cp.Variable((n_items, n_daughters), boolean=True)

# Total values for each daughter
V = [cp.sum(cp.multiply(values, x[:, j])) for j in range(n_daughters)]
V_bela, V_daniela, V_cassandra = V[0], V[1], V[2]

# Constraints
constraints = [cp.sum(x[i, :]) == 1 for i in range(n_items)]  # Each item to one daughter
constraints.append(V_cassandra >= V_daniela)
constraints.append(V_daniela >= V_bela)

# Objective: Minimize Cassandra's value minus Bela's
problem = cp.Problem(cp.Minimize(V_cassandra - V_bela), constraints)
problem.solve(solver=cp.CBC, verbose=True)

# Output results
print(f"Status: {problem.status}")
print(f"Minimal difference (Cassandra - Bela): {problem.value}")
print(f"Bela's total: {V_bela.value}")
print(f"Daniela's total: {V_daniela.value}")
print(f"Cassandra's total: {V_cassandra.value}")

# Print assignments (example output)
for i in range(n_items):
    for j in range(n_daughters):
        if x[i, j].value > 0.5:
            print(f"Item {i+1} ({values[i]}) -> {'Bela' if j==0 else 'Daniela' if j==1 else 'Cassandra'}")