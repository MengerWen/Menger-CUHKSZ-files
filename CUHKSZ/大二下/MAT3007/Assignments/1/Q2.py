import cvxpy as cp

# Decision variables
x1 = cp.Variable(nonneg=True)  # Number of product I
x2 = cp.Variable(nonneg=True)  # Number of product II
x3 = cp.Variable(nonneg=True)  # Number of product III

# Objective function (net profit)
profit = 57*x1 + 54*x2 + 80*x3

# Constraints
assembly_labor_constraint = (1/3)*x1 + (1/4)*x2 + (1/2)*x3 <= 150
testing_labor_constraint = (1/5)*x1 + (1/4)*x2 + (1/4)*x3 <= 80

# Problem definition
problem = cp.Problem(cp.Maximize(profit), [assembly_labor_constraint, testing_labor_constraint])

# Solve the problem
problem.solve()

# Results
print(f"Optimal number of product I: {x1.value:.2f}")
print(f"Optimal number of product II: {x2.value:.2f}")
print(f"Optimal number of product III: {x3.value:.2f}")
print(f"Maximum Net Profit: {problem.value:.2f}")