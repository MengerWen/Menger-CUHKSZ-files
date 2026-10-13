from scipy.optimize import linprog

# Primal problem
c_primal = [5, 4]  # Coefficients of objective function
A_ub_primal = [[-3, -1], [-2, -4]]  # Coefficients of constraints (≤ form)
b_ub_primal = [-30, -40]  # Right-hand side of constraints

result_primal = linprog(c_primal, A_ub=A_ub_primal, b_ub=b_ub_primal)
primal_solution = result_primal.x
primal_value = result_primal.fun

print("Primal Optimal Solution (x1, x2):", primal_solution)
print("Primal Optimal Value (Minimum Cost):", primal_value)

# Dual problem
c_dual = [-30, -40]  # Minimize -(30y1 + 40y2)
A_ub_dual = [[3, 2], [1, 4]]  # Constraints (already ≤)
b_ub_dual = [5, 4]  # Right-hand side

result_dual = linprog(c_dual, A_ub=A_ub_dual, b_ub=b_ub_dual)
dual_solution = result_dual.x
dual_value = -result_dual.fun  # Negate to get maximized value

print("Dual Optimal Solution (y1, y2):", dual_solution)
print("Dual Optimal Value (Maximum Value):", dual_value)

# Verify Strong Duality
difference = primal_value - dual_value
print("Difference (Primal - Dual):", difference)
if abs(difference) < 1e-10:
    print("Strong Duality Theorem is satisfied (difference ≈ 0)")
else:
    print("Strong Duality Theorem may not hold (check computations)")