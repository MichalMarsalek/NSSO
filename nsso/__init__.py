"""
Neat syntax for scipy.optimize
Expression based definition of optimization problems.
"""

import numpy as np
import scipy.optimize as so

FUNCTIONS = {
    "u+": (lambda x: +x),
    "u-": (lambda x: -x),
    "+": (lambda x,y: x+y),
    "-": (lambda x,y: x-y),
    "*": (lambda x,y: x*y),
    "/": (lambda x,y: x/y),
    "@": (lambda x,y: x@y),
    "**": (lambda x,y: x**y),
    "T": (lambda x: x.T),
    "[]": (lambda x,slice: x[slice])
}

class Node:
    """Abstract class representing a node a tree."""
    __array_priority__ = 100
    def collect_variables(self, output):
        """
        Recursively collects all variables occuring in the tree.
        Args:
            output (set): All variables found are added to this set.
        """
        raise NotImplementedError("Abstract method")

    def __repr__(self):
        raise NotImplementedError("Abstract method")

    def __str__(self):
        raise NotImplementedError("Abstract method")

class Expression(Node):
    """
    Tree representation of an expression.
    This expression can be used as an objective function or in a constraint.
    """

    def evaluate(self, values):
        """
        Performs a substitution of values into variables.

        Args:
            values Dict[Var, Array]: Values to be substituted.

        Returns:
            Array: Evaluated expression.

        """
        raise NotImplementedError("Abstract method")

    def __add__(self, other):
        return Func("+", self, other)
    def __pos__(self):
        return Func("u+", self)
    def __neg__(self):
        return Func("u-", self)
    def __radd__(self, other):
        return Func("+", other, self)
    def __sub__(self, other):
        return Func("-", self, other)
    def __rsub__(self, other):
        return Func("-", other, self)
    def __mul__(self, other):
        return Func("*", self, other)
    def __rmul__(self, other):
        return Func("*", other, self)
    def __matmul__(self, other):
        return Func("@", self, other)
    def __rmatmul__(self, other):
        return Func("@", other, self)
    def __div__(self, other):
        return Func("/", self, other)
    def __rdiv__(self, other):
        return Func("/", other, self)
    def __pow__(self, other):
        return Func("**", self, other)
    def __rpow__(self, other):
        return Func("**", other, self)

    def __getitem__(self, slice):
        return Func("[]", self, slice)

    def __eq__(self, other):
        return Constraint("==", self, other)
    def __le__(self, other):
        return Constraint("<=", self, other)
    def __ge__(self, other):
        return Constraint(">=", self, other)

    @property
    def T(self):
        """
        Transposition.

        Returns:
            Transposed expression.
        """
        return Func("T", self)

    def minimize(self, constraints=[], *args, **kwargs):
        """
        Problem minimization.

        Args:
            constraints(List[Constraint]): Constraints.

        Returns:
            Solution to the problem of minimizing the value of self, subject to give constraints.
        """
        return Problem(self, constraints).minimize(*args, **kwargs)

    def maximize(self, constraints=[], *args, **kwargs):
        """
        Problem maximization.

        Args:
            constraints(List[Constraint]): Constraints.

        Returns:
            Solution to the problem of maximizing the value of self, subject to give constraints.
        """
        return Problem(self, constraints).maximize(*args, **kwargs)

class Func(Expression):
    """
    Function node.
    This is any nonleaf node in the tree - including unary and binary operations.

    Attributes:
        func (Function): Function to be called.
        args (List[Expression]): Arguments (child nodes).
    """
    def __init__(self, func, *args):
        if isinstance(func, str):
            self.name = func
            self.func = FUNCTIONS[func]
        else:
            self.name = repr(func)
            self.func = func
        self.args = [x if isinstance(x, Expression) else Const(x) for x in args]

    def collect_variables(self, output):
        for son in self.args:
            son.collect_variables(output)

    def evaluate(self, values):
        return self.func(*(x.evaluate(values) for x in self.args))

    def __repr__(self):
        return f"Func({self.name}, " + ", ".join(repr(x) for x in self.args) + ")"

    def __str__(self):
        return f"{self.name}(" + ", ".join(str(x) for x in self.args) + ")"

class Const(Expression):
    """
    Constant leaf node.

    Attributes:
        value (Array): Value of the node.
    """
    def __init__(self, value):
        self.value = value

    def collect_variables(self, output):
        pass

    def evaluate(self, values):
        return self.value

    def __repr__(self):
        return f"Const({self.value})"

    def __str__(self):
        return f"{self.value}"

class Var(Expression):
    """
    Variable leaf node.

    Attributes:
        guess (Array): Initial guess.

    Properties:
        flat_guess (Array): Initial guess converted to a column vector.
        shape (Tuple[int, int]): Shape of variable.
        size (int): Size of the variable.
        is_row (bool): Indicator of row/column vector.
    """

    def __init__(self, guess=1, name="noname"):
        if isinstance(guess, int):
            self.guess = np.zeros((guess,))
        elif isinstance(guess, tuple):
            self.guess = np.zeros(guess)
        else:
            self.guess = np.array(guess)
        self.name = name

    @property
    def flat_guess(self):
        return np.resize(self.guess, (self.size,1))

    @property
    def shape(self):
        return self.guess.shape

    @property
    def size(self):
        return self.guess.size

    @property
    def is_row(self):
        return self.guess.ndim > 1 and self.shape[1] > 1

    def collect_variables(self, output):
        output.add(self)

    def evaluate(self, values):
        return values[self]

    def __repr__(self):
        return f"Var({self.name})"
    def __str__(self):
        return f"{self.name}"
    def __hash__(self):
        return id(self)

class Constraint(Node):
    """
    Constraint node.

    Attributes:
        type (string): "<=", ">=" or "=="
        left (Expression): Left size of the relation.
        right (Expression): Right size of the relation.

    Properties:
        type2 (string): For normalized constraints maps "==" to "eq" and "<=" to "ineq"
    """
    def __init__(self, type, left, right):
        self.type = type
        self.left = left if isinstance(left, Node) else Const(left)
        self.right = right if isinstance(right, Node) else Const(right)
    def collect_variables(self, output):
        self.left.collect_variables(output)
        self.right.collect_variables(output)

    def __repr__(self):
        return repr(self.left) + " " + self.type + " " + repr(self.right)

    def __str__(self):
        return str(self.left) + " " + self.type + " " + str(self.right)

    @property
    def type2(self):
        return {"==":"eq", "<=":"ineq"}[self.type]

    def normalized(self):
        """
        Converts a constraint to Expression <= 0 or Expression == 0 form.
        Returns:
            Constraint: normalized constraint.
        """
        if self.type == ">=":
            return Constraint("<=", self.right, self.left).normalized()
        return Constraint(self.type, self.left - self.right, Const(0))

class Problem:
    """
    Description of the problem.

    Attributes:
        objective (Expression): Objective function.
        constraitns (List[Constraint]): Constraints.
        variables (List[Var]): Variables in the problem.
        var_slices (Dict[Var, Slice]): Mapping from variables to slices in the combined vector.
    """
    def __init__(self, objective, constraints=[]):
        self.objective = objective
        self.constraints = constraints
        self._collect_vars()

    def _collect_vars(self):
        """
        Collects all variables and allocates positions (slices) for them.
        """
        res = set()
        self.objective.collect_variables(res)
        for c in self.constraints:
            c.collect_variables(res)
        self.variables = list(res)
        self.var_slices = {}
        start = 0
        for var in self.variables:
            self.var_slices[var] = slice(start, start + var.size)
            start += var.size

    def _transform(self, expr):
        """
        Transforms an expression to a function.
        Args:
            expr (Expression): Expression to be transformed.

        Returns:
            A function from variables to constants.
        """
        def transformed(x):
            args = {}
            for var, slice in self.var_slices.items():
                args[var] = x[slice].T if var.is_row else x[slice]
            return expr.evaluate(args)
        return transformed

    def minimize(self, *args, **kwargs):
        """
        Minimization of a problem.
        Prepares data for scipy.optimize, runs it a then slices and maps the results back to the Variables.
        """
        self._collect_vars()
        objective = self._transform(self.objective)
        constraints = []
        for c in self.constraints:
            c = c.normalized()
            constraints.append({"type": c.type2, "fun":self._transform(c.left)})
        guess = np.concatenate([var.flat_guess for var in self.variables])
        optim = so.minimize(objective, guess, *args, constraints=constraints, **kwargs)
        res = {}
        for var, slice in self.var_slices.items():
            res[var] = optim.x[slice].T if var.is_row else optim.x[slice]
        return res

    def maximize(self):
        """
        Maximization of a problem.
        """
        return Problem(-self.objective, self.constraints).minimize()

#other expression friendly functions

def norm_squared(arg):
    return Func(lambda x: np.sum(x**2), arg)

def norm(arg):
    return Func(lambda x: np.norm(x), arg)