#!/bin/env python

from symengine import Symbol, Matrix
import symengine as se

def approximate_integers(expr, tolerance=1e-6):
    """
    Recursively convert almost integer floats in a SymEngine expression to integers.
    """
    if isinstance(expr, se.Float):
        # Check if the float is close to an integer
        expr_int = round(float(expr))
        if abs(float(expr) - expr_int) < tolerance: 
            return se.Integer(expr_int)
        else:
            return expr
    elif isinstance(expr, se.Add) or isinstance(expr, se.Mul):
        # Apply conversion to each argument of the Add or Mul expression
        return expr.func(*[approximate_integers(arg, tolerance) for arg in expr.args])
    elif isinstance(expr, se.Pow):
        # Apply conversion to the base and exponent
        return se.Pow(approximate_integers(expr.args[0], tolerance), approximate_integers(expr.args[1], tolerance))
    elif isinstance(expr, se.Function):
        # Apply conversion to each argument of the function
        return expr.func(*[approximate_integers(arg, tolerance) for arg in expr.args])
    elif isinstance(expr, se.Matrix):
        return se.Matrix([approximate_integers(e, tolerance) for e in expr]).reshape(expr.rows, expr.cols)
    else:
        return expr

def subs_with_indexed(expr : se.Expr, q : se.Matrix, name : str):
    '''
    Subsitute variables in q with name_{i}
    '''
    dq = se.diff(q, 't')
    ddq = se.diff(dq, 't')

    
    # it's important to replace the highest derivatives first, as otherwise Derivative(q(t), t) will turn into 0
    return expr.subs(
        { ddqi: Symbol(f'dd{name}_{i}') for i,ddqi in enumerate(ddq) }
    ).subs(
        { dqi: Symbol(f'd{name}_{i}') for i,dqi in enumerate(dq) }
    ).subs(
        { qi: Symbol(f'{name}_{i}') for i,qi in enumerate(q) }
    )
    
def flatten_matrix(matrix):
    '''
    Flatten matrix in col-major format e.g. for use with Eigen.
    '''
    # Get dimensions of the original matrix
    rows, cols = matrix.shape
    
    for col in range(cols):
        for row in range(rows):
            yield matrix[row, col]