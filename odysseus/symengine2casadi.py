
import symengine as se
import casadi as ca

def to_casadi(expr : se.Expr, subs : dict):
    '''
    Converts symengine expression(s) into casadi expression(s).
    '''

    # This works if the expressions are identical, they don't have to be the same Object.
    if expr in subs:
        return subs[expr]

    elif isinstance(expr, se.Matrix):

        ca_exprs = [to_casadi(elem, subs) for elem in expr]
        
        return ca.vertcat(*ca_exprs).reshape((expr.rows, expr.cols))

    elif isinstance(expr, se.Add):

        res = 0

        for arg in expr.args:
            res += to_casadi(arg, subs)

        return res

    elif isinstance(expr, se.Mul):

        res = 1

        for arg in expr.args:
            res *= to_casadi(arg, subs)
            
        return res

    elif isinstance(expr, se.Pow):
        
        return ca.power(to_casadi(expr.args[0], subs), to_casadi(expr.args[1], subs))

    elif isinstance(expr, se.Function):
    
        func_name = type(expr).__name__

        if hasattr(ca, func_name):

            casadi_func = getattr(ca, type(expr).__name__)
            
            return casadi_func(*[to_casadi(arg, subs) for arg in expr.args])

    elif isinstance(expr, se.Number):

        return float(expr)

    elif expr is se.pi:

        return ca.pi

    
    raise Exception(f'Cannot convert expression of type {type(expr)} to casadi expression: \n{expr}')
