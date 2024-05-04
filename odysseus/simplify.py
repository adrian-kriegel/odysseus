
import symengine as se
import threading


def simplify_multi(exprs : list[se.Expr | se.Matrix]):
    '''
    Simplify expressions with one Thread per expression.
    Matrices count as multiple expressions.
    '''
    res = [None]*len(exprs)

    threads = []

    def simplify_into(i, expr : se.Expr):

        res[i] = expr.simplify()

    for i, expr in enumerate(exprs):

        if isinstance(expr, se.Matrix):
            r = expr.rows
            c = expr.cols
            res[i] = se.Matrix(simplify_multi([x for x in expr])).reshape(r, c)

        else:
            thread = threading.Thread(target=simplify_into, args=(i, expr))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    return res
            