# NSSO
Neater syntax for scipy.optimize

Since we weren't able to force CVXPY to find a local extreme of a nonconvex function for us, we were forced to use scipy.optimize.
However describing a optimization problem in the scipy.optimize framework is somewhat cumberstone, and so we came up with the idea to create a wrapper around it that would allow a user to describe a problem in a fashion similar to CVXPY syntax and then have it solved by scipy.optimize.

This is a functional proof of concept. It might be extended by supporting more types of expresions and functions based on what is needed for practice.
Also, symbolic calcuations of derivates could be added. Automatically computed gradients and Hess matrices could then be passed to the scipy.optimize module.

## Defining and solving an optimization problem in NSSO
We start by defining all the variables we will be using:
```
x = Var()                   # scalar variable
v = Var(4)                  # column vector of size 4 variable
z = Var(numpy_array, "z")   # variable with an initial guess (which determines the size) and a string name used for printing
```
Then, we can use these variables to form expressions. Such an expression can be used as an objective function:
```
objective = v.T @ z
```
Two expressions can be combined using `<=`, `>=` or `==` to form a constraint:
```
constraints = [x[0] >= v[1]]
```
A problem is then defined by a pair (objective, constraints) and solved:
```
problem = Problem(objective, constraints)
solution = problem.minimize()
```
An alternative way is:
```
solution = objective.minimize(constraints)
```



## Expressions
Common scalar/vector/matrix operations: `+`, `-`, `\*`, `/`, `@`, `\*\*`, `.T`, `[]` are available.
We can include the evaluation of any Python function using the `Func` object. For example
```
objective = Func(sin, x[0])         # -> is evaluated as sin(x[0])
```
For convenience, one can create a factory for common functions. For example, in the module this is done for `norm_squared` and `norm`.

```
def norm_squared(arg):
    return Func(lambda x: np.sum(x**2), arg)
```
