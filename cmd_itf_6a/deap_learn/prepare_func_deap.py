import numpy as np
from deap import gp

##################################### Overide the gp.PrimitiveTree
class MyGPTree(gp.PrimitiveTree):
    def __init__(self, content, op=str):
        gp.PrimitiveTree.__init__(self, content)
        self.op = op

    def __hash__(self):
        return hash(self.op(self))

    def __eq__(self, other):
        return self.op(self) == self.op(other)


def remove_twins(pop):
    return list(set(pop))

##################################### Prepare funcset
def prepare_funcset(strat, arity):

    #
    # Operator definition
    #
    def _div(left, right):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.divide(left, right)
            if isinstance(x, np.ndarray):
                x[np.isinf(x)] = 0
                x[np.isnan(x)] = 0
            elif np.isinf(x) or np.isnan(x):
                x = 0
        return x

    def _pow(y, n=1):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.power(y, n)
            if isinstance(x, np.ndarray):
                x[np.isinf(x)] = 0
                x[np.isnan(x)] = 0
            elif np.isinf(x) or np.isnan(x):
                x = 0
        return x

    def _log(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.log(x)
            if isinstance(x, np.ndarray):
                x[np.isinf(x)] = 1
                x[np.isnan(x)] = 1
            elif np.isinf(x) or np.isnan(x):
                x = 1
        return x

    def _sqrt(x):
        with np.errstate(invalid='ignore'):
            x = np.sqrt(x)
            if isinstance(x, np.ndarray):
                x[np.isinf(x)] = 0
                x[np.isnan(x)] = 0
            elif np.isinf(x) or np.isnan(x):
                x = 1
        return x

    funcset = gp.PrimitiveSet("MAIN", arity)
    funcset.addPrimitive(np.add, 2, name="Add")
    funcset.addPrimitive(np.subtract, 2, name="Sub")
    funcset.addPrimitive(np.multiply, 2, name="Mul")
    funcset.addPrimitive(_div, 2, name="Div")

    if "symc" in strat:
        id_ = len([x for x in dir(gp) if "symc" in x])  # number of instances of symc Ephemeral class
        funcset.addEphemeralConstant("symc{}".format(id_), lambda: 1.0)
        # funcset.addEphemeralConstant("symc{}".format(id_), lambda: random.randint(-1,1))

    if "pot" in strat:
        funcset.addPrimitive(np.square, 1, name="square")
        #funcset.addPrimitive(_sqrt, 1, name="sqrt")
    if "exp" in strat:
        funcset.addPrimitive(np.exp, 1, name="exp")
        funcset.addPrimitive(_log, 1, name="log")
    if "trigo" in strat:
        funcset.addPrimitive(np.sin, 1, name="sin")
        funcset.addPrimitive(np.cos, 1, name="cos")

    return funcset

############################## Constant Optimize
import scipy.optimize as opt
import copy


def generate_context(pset, data):
    context = {arg: dat for arg, dat in zip(pset.arguments, data.T)}
    context.update(pset.context)

    return context


def optimize_constants(ind, cost, context, precision=3, options=None, constraints=None):
    """ Update the constant values of ind according to:
    vec(c) = argmin_c ||yhat(data,c) - y||

    This needs to be called together when using symbolic constants.
    It may be called as a mutation operator together with the usage of ercs.
    """
    idx = [index for index, node in enumerate(ind) if isinstance(node, gp.Ephemeral)]

    if len(idx) == 0:
        return ind

    values = [ind[i].value for i in idx]
    values_bak = copy.copy(values)
    args = [("c%i" % i) for i in range(len(idx))]

    code = str(ind)
    for i, arg in zip(idx, args):
        code = code.replace(ind[i].format(), arg, 1)
    code = "lambda {args}: {code}".format(args=",".join(args), code=code)
    yhat = eval(code, context, {})
    with np.errstate(invalid='ignore', over='ignore'):
        res = opt.minimize(cost, values, args=yhat, options=options, constraints=constraints)

    if res.success and all(np.isfinite(res.x)):
        values = res.x
    else:
        values = values_bak

    for i, value in zip(idx, values):
        ind[i] = type(ind[i])()
        ind[i].value = round(value, 3)

    return ind


########################## simplify the expression
import re
from sympy import simplify, lambdify


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]

    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """

    prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)  # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


def simplify_this(expr):
    return simplify(stringify_for_sympy(expr))
