

import symengine as se

from odysseus.simplify import simplify_multi

def test_simplify_multi():

    x = se.Symbol('x')

    exprs = [
        se.sin(x)**2 + se.cos(x)*(1 + se.cos(x)),
        se.Matrix(
            [se.sin(x), -se.cos(x)]
        ).transpose() * se.Matrix(
            [se.sin(x), -se.cos(x)]
        ),
        se.Matrix(
            [
                [se.sin(x), 0],
                [1, -se.cos(x)]
            ]
        ) * se.Matrix(
            [
                [se.tan(x), 0],
                [1, -se.cos(x)]
            ]
        )
    ]

    simplified = simplify_multi(exprs)

    for orig, simp in zip(exprs, simplified):

        assert type(orig) == type(simp)

        if isinstance(orig, se.Matrix):
            for x,y in zip(orig, simp):
                assert x.simplify() == y
        else:
            assert orig.simplify() == simp