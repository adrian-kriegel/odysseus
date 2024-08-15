
import casadi as ca

import numpy as np

import matplotlib.pyplot as plt

from symengine import Matrix, Symbol, symbols
import symengine as se

from odysseus import Link, Transform

from test.rrbot import model_rrbot

from generate_nmpc import generate_pl_nmpc

from save import save_symengine, load_symengine

# axle_offset is the pivot point relative to the end of each rod
axle_offset = 0.05

# we don't care about the actual length of the segments, but the lenths between the axles
l0 = 1 - axle_offset*2
l1 = 1 - axle_offset*2

# mass
m0 = 1
m1 = 1

# damping
d0 = 0.7
d1 = 0.7

j0 = 0.0841666
j1 = 0.0841666


model = model_rrbot(Matrix([0,0,0]), Matrix([0,0,0]),l0,l1,m0,m1,j0,j1,d0,d1)

# add a virtual end-effector to set the goal pose
end_effector = Link('ef', model.find('segment1'), Transform(Matrix([0,0,l1])))

# we leave joint1 unactuated, so it comes first in the configuration space order
joints = [model.find_joint('joint1'), model.find_joint('joint0')]

n_active = 1
n_passive = 1

print('Actuated joint: joint0')

mass, rest = model.canonical_dynamics(joints)

subs_q_ext = { Symbol('g'): 9.81 }

mass = mass.subs(subs_q_ext)
rest = rest.subs(subs_q_ext)

save_symengine(mass, open('./mass.sym.pkl', 'wb'))
save_symengine(rest, open('./rest.sym.pkl', 'wb'))

assert load_symengine(open('./mass.sym.pkl', 'rb')) == mass
assert load_symengine(open('./rest.sym.pkl', 'rb')) == rest

model.find_segment('joint1')

target_position = Matrix(symbols('t_x t_y t_z'))

robot_position = end_effector.get_transform().trans_

weight = se.eye(3)

position_error = robot_position - target_position

objective = position_error.dot(weight * position_error)

q = model.q(joints)

solve = generate_pl_nmpc(
    q,
    mass,
    rest,
    n_passive=n_passive,
    state_constraint=Matrix([]),
    state_objective=objective,
    ocp_parameters=list(target_position),
    extra_substitutions=subs_q_ext,
    horizon = 1.0
)

solve.generate_dependencies('solver.c')

q0,q1 = se.symbols('q0 q1')
calc_position = se.lambdify([q0, q1], robot_position.subs({ q[0]: q0, q[1]: q1  }))

# state space dimension
n = len(q)*2
print("N", n)
nu = 1
steps = 20

# initial values of the decision variable
x0 = [0]*(steps*(n+nu))

n_constraints = n*steps
# continuity constraints are equality constraints, so lb == ub == 0
lbg = [0]*n_constraints
ubg = [0]*n_constraints

initial_state = [np.pi, 0] + [0.00]*(n-2)
target_position = [0,0,l1+l0]
print(initial_state + target_position)
res = solve(
    # initial guess for the decision variables
    x0 = x0,
    # initial state of the dynamic system
    p = initial_state + target_position,
    lbx = [-np.pi]*len(x0),
    ubx = [np.pi]*len(x0),
    lbg = lbg,
    ubg = ubg,
)

x = list(float(x) for x in [0] + initial_state + ca.vertsplit(res['x'], 1))

x = np.array(x).reshape(steps + 1, n+nu).transpose()
t = np.arange(steps + 1)

pos = np.array([calc_position(state[2], state[1]) for state in x.transpose()]).transpose()

plt.plot(t, x[0,:], label='u')
#plt.plot(t, x[1,:], label='x')
#plt.plot(t, x[2,:], label='y')
plt.plot(t, x[1,:], label='q1')
plt.plot(t, x[2,:], label='q0')
plt.plot(t, pos[0,:], label='x')
plt.plot(t, pos[2,:], label='z')
#plt.plot(t, x[4,:], label='dx')
#plt.plot(t, x[5,:], label='dy')
#plt.plot(t, x[6,:], label='q')
#plt.plot(t, x[7,:], label='v')
#plt.plot(t, x[8,:], label='a')

plt.legend()

plt.show()

