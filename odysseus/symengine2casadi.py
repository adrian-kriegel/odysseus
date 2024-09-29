
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
        
        # casadi's to_casadi(exp(x)) decays to (2.7...)^x otherwise
        if expr.args[0] is se.E:
            return ca.exp(to_casadi(expr.args[1], subs))

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

    elif expr is se.E:

        return ca.exp(1)

    
    raise Exception(f'Cannot convert expression of type {type(expr)} to casadi expression: \n{expr}')


class SymEngineToCasADiConverter:
    '''
    Helper class for converting symengine expressions to CasADi expressions.
    '''

    def __init__(self, subs : list[tuple[se.Expr, ca.MX]]):
        '''
        Arguments:
            subs: A list of tuples of the form (symengine expression, casadi expression).
        '''

        self.subs_dict_ = {}

        for se_expr, ca_expr in subs:

            ca_columns = ca_expr.columns() if isinstance(ca_expr, ca.MX) else 1
            ca_rows = ca_expr.rows() if isinstance(ca_expr, ca.MX) else 1
            se_rows = len(se_expr) if isinstance(se_expr, se.Matrix) else 1

            if ca_columns != 1:
                print(ca_expr.rows)
                raise Exception(f'Matrix substitution is not supported: {se_expr} -> {ca_expr}.')


            if ca_rows != se_rows:
                raise Exception(f'Mismatch in dimensions of {se_expr} ({se_rows} rows) and {ca_expr} ({ca_rows} rows).')

            # Substitute scalar expressions.
            if ca_rows == 1:
                self.subs_dict_[se_expr] = ca_expr
                continue

            # Substitute vector expressions.
            for se_subexpr, ca_subexpr in zip(se_expr, ca.vertsplit(ca_expr, 1)):

                if se_subexpr is se_expr:
                    raise Exception(f'Found duplicate substitution for {se_expr} in {subs}.')

                self.subs_dict_[se_subexpr] = ca_subexpr

    def __call__(self, expr : se.Expr):
        '''
        Converts symengine expression(s) into casadi expression(s).
        '''

        return to_casadi(expr, self.subs_dict_)