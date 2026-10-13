import pulp as lp
import matplotlib.pyplot as plt

# Data points
D = [
    [0, 8],
    [2, 0],
    [10, 3],
    [12, 10],
    [7, 12],
    [3, 12]
]

taus = [0.1, 1, 10]
solutions = []

for tau in taus:
    # Create the problem
    prob = lp.LpProblem(f"SparseRobustRegression_tau{tau}", lp.LpMinimize)
    
    # Variables
    x1 = lp.LpVariable('x1', cat='Continuous')
    x2 = lp.LpVariable('x2', cat='Continuous')
    y = [lp.LpVariable(f'y{i}', lowBound=0) for i in range(6)]
    z1 = lp.LpVariable('z1', lowBound=0)
    z2 = lp.LpVariable('z2', lowBound=0)
    
    # Objective: minimize sum of absolute residuals
    prob += lp.lpSum(y)
    
    # Constraints for residuals y_i
    for i in range(6):
        u_i, v_i = D[i]
        prob += y[i] >= u_i * x1 + x2 - v_i  # y_i >= a_i^T x - b_i
        prob += y[i] >= -(u_i * x1 + x2) + v_i  # y_i >= -a_i^T x + b_i
    
    # Constraints for z_j (L1 norm)
    prob += z1 >= x1
    prob += z1 >= -x1
    prob += z2 >= x2
    prob += z2 >= -x2
    prob += z1 + z2 <= tau  # L1 constraint
    
    # Solve the problem
    prob.solve(lp.PULP_CBC_CMD(msg=False))
    
    # Store solution if optimal
    if prob.status == lp.LpStatusOptimal:
        x1_val = x1.value()
        x2_val = x2.value()
        solutions.append((tau, x1_val, x2_val))
    else:
        solutions.append((tau, None, None))

# Plotting the results
plt.figure(figsize=(10, 6))
u_data = [point[0] for point in D]
v_data = [point[1] for point in D]
plt.scatter(u_data, v_data, color='black', label='Data points', zorder=3)

# Generate lines for each tau
u_vals = [0, 12]  # Extend from minimum to maximum u
for tau, x1, x2 in solutions:
    if x1 is None or x2 is None:
        continue
    v_fit = [x1 * u + x2 for u in u_vals]
    plt.plot(u_vals, v_fit, label=f'τ = {tau}', linewidth=2)

plt.xlabel('u')
plt.ylabel('v')
plt.title('Sparse Robust Regression Fit for Different τ Values')
plt.legend()
plt.grid(True)
plt.show()