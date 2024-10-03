#!/usr/bin/env python

'''
Disclaimer: This is a WIP. I haven't verified the resulting OCP solver yet.

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
from odysseus.examples.inject_metadata import create_metadata_getter, create_metadata_file

# TODO: read from inputs
end_effector_names = ['camera_link']
joints_actuated_names = ['joint1', 'joint2']
horizon = 1.0
num_shooting_intervals = 10

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
joints_free = []
joints = joints_actuated + joints_free

q = robot.q(joints_actuated)
dq = diff(q, 't')

end_effector_poses = [
  link.get_transform() for link in end_effectors
]

end_effector_positions = [
  transform.trans_ for transform in end_effector_poses
]

def forward_end_effector_positions(qv):
  return [pos.subs({ qi: v for qi, v in zip(q, qv) }) for pos in end_effector_positions]

print(forward_end_effector_positions([1.54773, 1.0285]))

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
ca_state = ca.vertcat(ca_q, ca_dq, ca_ddq)
model.x = ca_state

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


#
# Path cost.
#

ocp.cost.cost_type = 'NONLINEAR_LS'

ocp.model.cost_y_expr = ca.vertcat(
  ca_u, # Penalize high jerk
  ca_ddq, # Penalize high acceleration (approximately penalize hight torque)
  ca_end_effector_positions_joined
)

# Weight matrix *can* be changed at run-time.
ocp.cost.W = np.diag(
  # Jerk penalty.
  [0.005]*num_joints +
  # Acceleration penalty.
  [0.005]*num_joints +
  # Position error penalty.
  [1.0]*ca_end_effector_positions_joined.rows()
)

# Reference output (populated at run-time).
ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.rows())

#
# Terminal cost.
#

ocp.cost.cost_type_e = 'NONLINEAR_LS'

ocp.model.cost_y_expr_e = ca_end_effector_positions_joined

ocp.cost.W_e = np.identity(ocp.model.cost_y_expr_e.rows())

ocp.cost.yref_e = 3.0 * np.zeros(ocp.model.cost_y_expr_e.rows())

#
# Constraints 
#

# Constraints on the joints after t0.

# Acceleration is penalized in the cost anyway, so ~10 G as an upper bound is probably reasonable.
ddq_max = 100
# I have trouble imagining jerk values. A jerk of 20 m/s³ is apparently at the lower end of a car braking abruptly.
# We penalize high jerk anyway, so we can use a moderately high value for the constraint.
dddq_max = 20

# The joints don't have limits but it makes sense to constrain them to sensible ranges.
# This helps the solver by confining the search space.
q_min = - 2.0 * np.pi
q_max = 2.0 * np.pi
dq_max = 100

for joint in joints:
  if joint.limits_.pos_l_ == None:
    joint.limits_.pos_l_ = q_min
  if joint.limits_.pos_u_ == None:
    joint.limits_.pos_u_ = q_max
  if joint.limits_.vel_l_ == None:
    joint.limits_.vel_l_ = -dq_max
  if joint.limits_.vel_u_ == None:
    joint.limits_.vel_u_ = dq_max

ocp.constraints.lbx = np.array([j.limits_.pos_l_ for j in joints] + [j.limits_.vel_l_ for j in joints] + [-ddq_max]*len(joints))
ocp.constraints.ubx = np.array([j.limits_.pos_u_ for j in joints] + [j.limits_.vel_u_ for j in joints] + [ddq_max]*len(joints))
ocp.constraints.idxbx = np.arange(nx)

ocp.constraints.lbu = np.array([-dddq_max]*nu)
ocp.constraints.ubu = np.array([dddq_max]*nu)
ocp.constraints.idxbu = np.arange(nu)

# Initial state constraint. Populated at run-time.
ocp.constraints.lbx_0 = np.zeros(nx)
ocp.constraints.ubx_0 = np.zeros(nx)
ocp.constraints.idxbx_0 = np.arange(nx)

#
# Solver setup.
#

ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.qp_solver_cond_N = 1  # Condensing steps
ocp.solver_options.nlp_solver_type = 'SQP_RTI'

#
# Code generation.
#

tempdir = tempfile.mkdtemp()
code_dir = os.path.join(tempdir, model.name)

print('Generating code in', code_dir)

os.chdir(tempdir)

os.makedirs(code_dir, exist_ok=True)

ocp.code_export_directory = code_dir

builder = CMakeBuilder()

builder.options_on = [
  'BUILD_ACADOS_OCP_SOLVER_LIB',
]

builder.options_off = [
  'BUILD_EXAMPLE'
]

#
# Generate CMakeLists.txt and source code but don't build the shared library just yet.
#

try:
  ocp_solver = AcadosOcpSolver(
    ocp, 
    json_file='acados_ocp.json', 
    build=False,
    generate=True,
    cmake_builder=builder
  )

# ACADOS is a bit of a special child. It tries to load the shared library which we have told it to NOT generate...
except OSError as e:
  if not 'cannot open shared object file: No such file or directory' in str(e):
    raise e from None

#
# We now apply some patches to the generated code to inject metadata about the controller.
#

# Patch CMakeLists.txt
with open(os.path.join(code_dir, 'CMakeLists.txt'), 'r+') as f:
  cmakelists = f.read()

  # TODO: this heavily depends on things like formatting and the ACADOS version. 
  # It would be better to parse the CMakeLists.txt
  add_library_line = r"add_library(${LIB_ACADOS_OCP_SOLVER} SHARED $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${OCP_OBJ}>)"

  add_library_line_patched = r"add_library(${LIB_ACADOS_OCP_SOLVER} SHARED $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${OCP_OBJ}> controller_metadata.cpp)"

  if not add_library_line in cmakelists:

    raise Exception('Failed to patch CMakeLists.txt!')

  f.seek(0)
  f.write(cmakelists.replace(add_library_line, add_library_line_patched))

# Generate the metadata source file

# The order of the state interfaces needs to match the order in which they appear in the state vector.
state_interfaces = [
  joint.name() + '/position' for joint in joints
] + [
  joint.name() + '/velocity' for joint in joints
] + [
  joint.name() + '/acceleration' for joint in joints
]

command_interfaces = [
  joint.name() + '/position' for joint in joints
] + [
  joint.name() + '/velocity' for joint in joints
] + [
  joint.name() + '/acceleration' for joint in joints
]

reference_interfaces = [
  f'{link.name()}/position/{coord}' 
  for link in end_effectors 
  for coord in ['x', 'y', 'z'] 
]

print(reference_interfaces)

get_controller_metadata = create_metadata_getter(
  'get_controller_metadata', 
  {
    'joints': [joint.name() for joint in joints],
    'joints_actuated': [joint.name() for joint in joints_actuated],
    'joints_free': [joint.name() for joint in joints_free],
    'state_interfaces': state_interfaces,
    'command_interfaces': command_interfaces,
    'reference_interfaces': reference_interfaces
  }
)

get_acados_prefix = f'''
const char* get_acados_prefix() {{
  return "{model.name}"; 
}}
'''

metadata_code = create_metadata_file([get_controller_metadata, get_acados_prefix])

with open(os.path.join(code_dir, 'controller_metadata.cpp'), 'w') as f:

  f.write(metadata_code)

metadata_code = create_metadata_file([get_controller_metadata])

builder.exec(code_dir)
