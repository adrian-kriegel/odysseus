#!/bin/env python

'''
Derives the dynamics of the rrbot from the gazebo examples and outputs C code for Computed Torque Control (https://en.wikipedia.org/wiki/Computed_torque_control) which allows for exact linearization of nonlinear systems.

The code generation is a bit messy because it uses sympy, whereas Odysseus uses symengine. I strongly recommend that you use casadi for code generation instead. See odysseus.symengine2casadi.

'''

from symengine import Symbol, Matrix, Function
from sympy import Equality, MatrixSymbol, sympify, symbols
from sympy.utilities.codegen import codegen

from odysseus import diff, LinkModel, Segment, Link, JointRevolute, Transform, inertia_matrix

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

## build the model using Links/Segments

J0 = inertia_matrix(iyy=0.0841666)
J1 = inertia_matrix(iyy=0.0841666)

model = LinkModel()

# the root is fixed, so doesn't move
root = model.create_root('root', Matrix([0,0,0]), Matrix([0,0,0]))

joint0 = JointRevolute(
    'joint0',
    root,
    origin=Transform(),
    axis=Matrix([0,1,0]),
    damping=d0
)

segment0 = Segment('segment0', joint0, Transform(), m0, J0, inertial_offset=Matrix([0,0,l0/2]))

joint1 = JointRevolute(
    'joint1',
    segment0,
    origin=Transform(Matrix([0, 0, l0])),
    axis=Matrix([0,1,0]),
    damping=d1
)

segment1 = Segment('segment1', joint1, Transform(), m1, J1, inertial_offset=Matrix([0,0,l1/2]))

tau = model.dynamics()

## substitute q(t) and derivatives for symbols representable as c variables

param_q = symbols('q0 q1')
param_dq = symbols('dq0 dq1')
param_ddq = symbols('ddq0 ddq1')

q = [joint0.q(), joint1.q()]

t = Symbol('t')

for i in range(0, len(q)):

    tau = tau\
        .subs(diff(diff(q[i], t), t), param_ddq[i])\
        .subs(diff(q[i], t), param_dq[i])\
        .subs(q[i], param_q[i])

tau = tau.subs('g', 9.81)

outparam = MatrixSymbol('u_out', len(q), 1)

((_, code), _) = codegen(
    ('compute_torque', Equality(outparam, sympify(tau))), 
    argument_sequence=(
        outparam, 
        param_ddq[0], 
        param_ddq[1], 
        param_dq[0], 
        param_dq[1], 
        param_q[0], 
        param_q[1]
    ),
    language='C99'
)

print(code)