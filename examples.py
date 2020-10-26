import nsso as ns
import numpy as np
from math import sin

# Some examples from CVXPY website
# Generate a random non-trivial linear program.
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the problem.
x = ns.Var((n,1))
objective = c.T @ x
constraints = [A @ x <= b]
x = objective.minimize(constraints)
print(x)

# Generate data.
m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define and solve the problem.
x = ns.Var((n,))
cost = ns.norm_squared(A @ x - b)
x = cost.minimize()
print(x)