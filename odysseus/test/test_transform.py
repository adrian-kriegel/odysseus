
from sympy import Matrix, symbols, eye

from odysseus import Transform

def test_transform():

    transform = Transform(Matrix(symbols('x y z')), symbols('rx ry rz'))
    
    assert transform.trans_ == Matrix(symbols('x y z'))

    # TODO: more