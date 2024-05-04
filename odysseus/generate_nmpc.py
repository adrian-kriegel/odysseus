#!/bin/env python

import casadi as ca
import symengine as se

from symengine2casadi import to_casadi


def generate_ocp_solver(
    state,
    controls,
    ode,
    state_objective,
    ocp_parameters = [],
    horizon=1.0,
    steps=20
):
    '''
    Generate an OCP solver using multiple shooting method.

    '''

    dt = horizon/steps
    #
    # Initialzation of time dependant variables.
    # These will be filled up in the for loop iterating over the prediction horizon.
    #

    constraints_continuity = []     # continuity constraints for the solution
    
    constraints_user = [] # user-defined constraints

    objective = 0      # objective function

    def make_state_vector(k : int):
        return ca.MX.sym(f'x_{k}', state.rows())

    state_0 = make_state_vector(k=0)
    # state at current time tk in the for loop below
    state_k = state_0
    
    # these will be [u0 x1 u1 x2...]
    decision_vars = []

    # create NLP via multiple shooting
    for k in range(steps):
        # Control input at tk.
        controls_k = ca.MX.sym(f'u_{k}', controls.rows())
        decision_vars.append(controls_k)
        
        # Integrator for dx = f(x,u) over [tk,tk+dt].
        integrator = ca.integrator(
            f'F_{k}', 
            'rk', 
            {
                'x': state,
                'p': ca.vertcat(controls), 
                'ode': ode
            },
            0,
            dt,
            {
                # This is required for exporting C code.
                'simplify': True,
            }
        )
        
        integrator_res = integrator(x0=state_k, p=ca.vertcat(controls_k))

        state_k_next = integrator_res['xf']

        # Increment the "iterator" state_k.
        state_k = make_state_vector(k)
        
        decision_vars.append(state_k)
        
        # Continuity constraint (state_k has been reassigned).
        constraints_continuity.append(state_k_next - state_k)

        objective += state_objective(state_k)

    # combine all constraints
    constraints = constraints_user + constraints_continuity

    problem = {
        'f': objective,
        'x': ca.vertcat(*decision_vars),
        'g': ca.vertcat(*constraints),
        'p': ca.vertcat(state_0, ocp_parameters)
    }

    return ca.nlpsol('solver', 'ipopt', problem)


def generate_pl_nmpc(
    q : se.Expr,
    mass : se.Expr,
    rest : se.Expr,
    n_passive : int,
    state_objective : se.Expr,
    state_constraint : se.Matrix,
    ocp_parameters : list[se.Expr],
    extra_substitutions = {},
    **kwargs
):
    '''
    Generate OCP solver for NMPC of the partially linearized (pl) system.
    The system is assumed to be ordered such that the first n_passive coordinates are not directly actuated.
    '''

    n_active = len(q) - n_passive

    # partition the mass matrix according to q1, q2, lambda
    m11 = mass[0:n_passive, 0:n_passive]
    m12 = mass[0:n_passive, n_passive:len(q)]

    # notation from 'underactuated robotics' has the 'rest' terms on the right and calls it tau1 and tau2
    # so we need to add minus sign
    tau1 = - se.Matrix(rest[0:n_passive])

    # helper terms
    dq = se.diff(q, 't')
    ddq = se.diff(dq, 't')

    # coordinates of the directly acuated system
    ddq2 = se.Matrix(ddq[n_passive:])

    # we can model the dynamics of the passive part as an output from the active part ddq1 = f_out(...) fed into integrators
    # this actually only works if we haven't made too many approximations in the mass and inertia terms
    # assuming mass=0 or inertia=0 for any segments may lead to semi-definite m11
    ddq1 = m11.inv() * (tau1 - m12*ddq2)
    
    # coordinates of the passive joints

    ca_q1 = ca.MX.sym('q_1', n_passive)
    ca_dq1 = ca.MX.sym('dq_1', n_passive)


    # coordinates of the active joins
    ca_q2 = ca.MX.sym('q_2', n_active)
    ca_dq2 = ca.MX.sym('dq_2', n_active)

    # we're acting as if ca_ddq2 were a direct control input
    ca_ddq2 = ca.MX.sym('ddq_2', n_active)

    # substitutions to convert symengine expressions into casadi expressions
    def subs_array(exprs, ca_exprs):
        return { 
            x: ca_x for (x, ca_x) in zip(exprs, ca_exprs)
        }

    # convert parameters to casadi types
    ca_ocp_parameters = [ca.MX.sym(param.name) for param in ocp_parameters]

    subs = subs_array(
        dq[0:n_passive], ca.vertsplit(ca_dq1, 1)
    ) | subs_array(
        q[0:n_passive], ca.vertsplit(ca_q1, 1)
    ) | subs_array(
        ddq[n_passive:], ca.vertsplit(ca_ddq2, 1)
    ) | subs_array(
        dq[n_passive:], ca.vertsplit(ca_dq2, 1)
    ) | subs_array(
        q[n_passive:], ca.vertsplit(ca_q2, 1)
    ) | { param: ca_param for param, ca_param in zip(ocp_parameters, ca_ocp_parameters) } | extra_substitutions


    # perform cse before converting to casadi
    # cse is absolutely required as the solver will require too much memory otherwise
    subexpr_ddq1, ddq1 = se.cse(list(ddq1))

    # subexpressions may depend on other subexpressions so we can't just join dicts, but need to successively grow `subs`
    for x,expr in subexpr_ddq1:
        subs[x] = to_casadi(expr, subs)

    # ca_ddq1 is defined in terms of the coordinates of the directly actuated system
    ca_ddq1 = ca.vertcat(
        *[to_casadi(ddq1i, subs) for ddq1i in ddq1]
    )

    ode_state = ca.vertcat(
        # passive part
        ca_q1,
        ca_dq1,
        # actuated part
        ca_q2,
        ca_dq2
    )

    # dstate = f(state)
    ode = ca.vertcat(
        # d(ca_q1) = ca_dq1 (integrator)
        ca_dq1, 
        # d(ca_dq1) = ca_ddq1 = f_out(q2,dq2,ddq2) - this is where the magic happens
        ca_ddq1,
        # d(ca_q2) = ca_dq2 (integrator)
        ca_dq2, 
        # d(ca_dq2) = ca_ddq2 (integrator of the control input ca_ddq2)
        ca_ddq2, 
    )

    ca_state_objective = ca.Function(
        'cost', 
        [ode_state], 
        [to_casadi(state_objective, subs)], 
        { 'allow_free': True }
    )
    
    cost = ca.Function('cost_', [ode_state, *ca_ocp_parameters], [to_casadi(state_objective, subs)])

    print("cost", cost([0,0,0,0], 0,0,1))

    print('Generating solver...')

    return generate_ocp_solver(
        state=ode_state,
        controls=ca_ddq2,
        ode=ode,
        state_objective=ca_state_objective,
        ocp_parameters=ca.vertcat(*ca_ocp_parameters),
        **kwargs
    )
