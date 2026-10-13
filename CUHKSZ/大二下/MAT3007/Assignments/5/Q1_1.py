import cvxpy as cp

x12 = cp.Variable()
x13 = cp.Variable()
x23 = cp.Variable()
x24 = cp.Variable()
x34 = cp.Variable()
delta = cp.Variable()

constraints = [
    x23 + x24 - x12 == 0,
    x13 + x23 - x34 == 0,
    delta - x12 - x13 == 0,
    x24 + x34 - delta == 0,
    x12 >= 0, x12 <=8,
    x13 >= 0, x13 <= 7,
    x23 >= 0, x23 <= 2,
    x24 >= 0, x24 <= 4, 
    x34 >= 0, x34 <= 12
]

obj = delta

problem = cp.Problem(cp.Maximize(obj),constraints)
problem.solve()

print(f"x12 = {x12.value:.2f}")
print(f"x13 = {x13.value:.2f}")
print(f"x23 = {x23.value:.2f}")
print(f"x24 = {x24.value:.2f}")
print(f"x34 = {x34.value:.2f}")
print(f"optimal_value = {obj.value:.2f}")