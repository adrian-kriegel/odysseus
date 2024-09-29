#!/usr/bin/env python

'''
Loads a URDF file and generates an NMPC solver for inverse dynamics control.

The resulting controller assumes that the dynamics of the actuated joints are decoupled and linear.
This can be achieved by chaining a computed torque controller (see urdf_computed_torque.py) behind the NMPC controller.

The NMPC model assumes that the system is jerk-controlled, though the jerk is never passed to any lower-level controller. 
It is merely used to ensure that the resulting trajectory is three-times differentiable, which means that it can be exactly
followed via the [PID -> CT] stack.

Alternatively, we could model the acceleration as the input if we modelled the dynamics of the PID controller. 
This would probably lead to issues with the solver "taking advantage" of the oscillations (especially towards
the end of the horizon) if the cost function is not carefully chosen.

'''

import sys

import os

import tempfile

import casadi as ca

import numpy as np

from symengine import Matrix, Symbol

from odysseus import diff, LinkModel
from odysseus.symengine2casadi import SymEngineToCasADiConverter

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from acados_template.builders import CMakeBuilder

from odysseus.examples.model_from_args import urdf, get_robot_name, base_link_name

# TODO: read from inputs
end_effector_names = ['camera_link']
joints_actuated_names = ['joint1', 'joint2']
horizon = 0.25
num_shooting_intervals = 5

robot_name = get_robot_name()

# Create a link between the world and the element we want to load from the URDF.
world_to_base = urdf.model().create_root(
  name='world_to_base',
  xyz=Matrix([0,0,0]),
  rpy=Matrix([0,0,0])
)

robot : LinkModel = urdf.model()



# Find and load the element named `base_link_name` from the URDF.
# This will also load all child elements.
urdf.add_element_by_name(
  # Find the element named `base_link_name` in the URDF.
  base_link_name,
  # Add this link as a child of world_to_base.
  world_to_base
)


end_effectors = [robot.find(name) for name in end_effector_names]
joints_actuated = [robot.find_joint(name) for name in joints_actuated_names]

q = robot.q(joints_actuated)
dq = diff(q, 't')

end_effector_poses = [
    link.get_transform() for link in end_effectors
]

end_effector_positions = [
    transform.trans_ for transform in end_effector_poses
]

# Introduce CasADi variables for converting the model in the next step.

num_joints = len(joints_actuated)

# Joint positions.
ca_q = ca.MX.sym('q', num_joints)

# Joint velocities.
ca_dq = ca.MX.sym('dq', num_joints)

# Joint accelerations.
ca_ddq = ca.MX.sym('ddq', num_joints)

# For solving the OCP, we use as the jerk of the system as the input. 
ca_u = ca.MX.sym('u', num_joints)

to_casadi = SymEngineToCasADiConverter([
  (dq, ca_dq),
  (q, ca_q),
  (Symbol('g'), 9.81)
])

# Convert expressions to CasADi.

ca_end_effector_positions = [
  to_casadi(position) for position in end_effector_positions
]

ca_end_effector_positions_joined = ca.vertcat(*ca_end_effector_positions)

# Build an AcadosModel from the URDF.


model = AcadosModel()

model.name = f'model_{robot_name}'

# State vector (ddq becomes part of  the state because we model the system as being jerk-controlled).
model.x = ca.vertcat(ca_q, ca_dq, ca_ddq)

# TODO: is there a special Linear ODE model in ACADOS?
# Simple integrator chain.
model.f_expl_expr = ca.vertcat(ca_dq, ca_ddq, ca_u)
model.u = ca_u

# In case we want to use an implicit integrator.
model.xdot = ca.MX.sym('xdot', model.x.rows())
model.f_impl_expr = model.xdot - model.f_expl_expr

#
# OCP Formulation
#

ocp = AcadosOcp()

ocp.model = model

nx = model.x.rows()
nu = model.u.rows()

ocp.solver_options.tf = horizon
ocp.dims.N = num_shooting_intervals

# TODO: This should be 'LINEAR_LS'
ocp.cost.cost_type = 'NONLINEAR_LS'
ocp.model.cost_y_expr = ca_end_effector_positions_joined

# Weight matrix *can* be changed at run-time.
ocp.cost.W = np.identity(ocp.model.cost_y_expr.rows())

# Reference output (populated at run-time).
ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.rows())

#
# Constraints 
#

ocp.constraints.lbx = np.array([j.limits_.pos_l_ for j in joints_actuated] + [j.limits_.vel_l_ for j in joints_actuated])
ocp.constraints.ubx = np.array([j.limits_.pos_u_ for j in joints_actuated] + [j.limits_.vel_u_ for j in joints_actuated])
# TODO: Constrain the acceleration! 
ocp.constraints.idxbx = np.arange(len(joints_actuated)*2)

# TODO: Constraint the jerk!

#
# Solver setup.
#

ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.qp_solver_cond_N = 5  # Condensing steps
ocp.solver_options.nlp_solver_type = 'SQP_RTI'



#
# Code generation.
#

tempdir = '.acados_generated'
code_dir = os.path.join(tempdir, model.name)
#os.chdir(tempdir)
#os.makedirs(code_dir, exist_ok=True)

ocp.code_export_directory = code_dir

builder = CMakeBuilder()

builder.options_on = [
  'BUILD_ACADOS_SOLVER_LIB',
]

builder.options_off = [
  'BUILD_EXAMPLE'
]

ocp_solver = AcadosOcpSolver(
  ocp, 
  json_file='acados_ocp.json', 
  build=True,
  generate=True,
  cmake_builder=builder
)
