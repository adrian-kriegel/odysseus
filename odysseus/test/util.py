
# TODO: use symengine (trigsimp?) --> actually, symengine uses sympy for simplfication...
import sympy as sp

def assert_eq(a, b):

    assert sp.simplify(sp.trigsimp(a)) == sp.simplify(sp.trigsimp(b))

