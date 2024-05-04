
import pickle 

import symengine as se

def save_symengine(expr : se.Expr | se.Matrix, file):

    if isinstance(expr, se.Matrix):
        data = list(expr)
        shape = expr.shape
    else:
        data = [expr]
        shape = None

    return pickle.dump((shape, data), file, protocol=-1)

def load_symengine(file):

    shape, data = pickle.load(file)

    if shape is None:
        return data
    else:
        r,c = shape
        return se.Matrix(data).reshape(r, c)