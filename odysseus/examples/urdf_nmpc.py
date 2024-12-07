#!/usr/bin/env python

'''
Disclaimer: This is a WIP. I haven't verified the resulting OCP solver yet.

Loads a URDF file and generates an NMPC solver for inverse dynamics control.

The resulting controller assumes that the dynamics of the actuated joints are decoupled and linear.
This can be achieved by chaining a computed torque controller (see urdf_computed_torque.py) behind the NMPC controller.

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

def eval_end_effector_positions(qv):
  ''' Quick helper for debugging. Returns end effector positions in world coordinates for values of q. '''
  return [pos.subs({ qi: v for qi, v in zip(q, qv) }) for pos in end_effector_positions]


# Introduce CasADi variables for converting the model in the next step.

num_joints = len(joints_actuated)

# Joint positions.
ca_q = ca.MX.sym('q', num_joints)

# Joint velocities.
ca_dq = ca.MX.sym('dq', num_joints)

# Joint accelerations.
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

model = AcadosModel()

model.name = f'model_{robot_name}'

ca_state = ca.vertcat(ca_q, ca_dq)
model.x = ca_state

# TODO: is there a special Linear ODE model in ACADOS?
# Simple integrator chain.
model.f_expl_expr = ca.vertcat(ca_dq, ca_u)
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

end_effector_coords_names = [
  f'{link.name()}_{coord}/position' 
  for link in end_effectors 
  for coord in ['x', 'y', 'z'] 
]

# TODO: Tie this more closely to the cost expression.
path_reference_names = [
  f'{joint.name()}/acceleration' for joint in joints 
] + end_effector_coords_names

ocp.cost.cost_type = 'NONLINEAR_LS'

ocp.model.cost_y_expr = ca.vertcat(
  ca_u, # Penalize high acceleration (approximately penalize high torque)
  ca_end_effector_positions_joined
)

# Weight matrix *can* be changed at run-time.
ocp.cost.W = np.diag(
  # Acceleration penalty.
  [0.01]*num_joints +
  # Position error penalty.
  [1.0]*ca_end_effector_positions_joined.rows()
)

# Reference output (populated at run-time).
ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.rows())

#
# Terminal cost.
#

# TODO: Tie this more closely to the cost expression.
terminal_reference_names = end_effector_coords_names

ocp.cost.cost_type_e = 'NONLINEAR_LS'

# Make sure this aligns with terminal_reference_names
ocp.model.cost_y_expr_e = ca_end_effector_positions_joined

ocp.cost.W_e = 3.0 * np.identity(ocp.model.cost_y_expr_e.rows())

ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.rows())

#
# Constraints 
#

# Constraints on the joints after t0.

# Acceleration is penalized in the cost anyway, so ~10 G as an upper bound is probably reasonable.
ddq_max = 100

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

# Somehow ACADOS does not allow us to use soft constraints on the state directly...
# But it's allowed on h(x) = x
#model.con_h_expr = model.x

#ocp.constraints.lh = np.array([j.limits_.pos_l_ for j in joints] + [j.limits_.vel_l_ for j in joints] + [-ddq_max]*len(joints))
#ocp.constraints.uh = np.array([j.limits_.pos_u_ for j in joints] + [j.limits_.vel_u_ for j in joints] + [ddq_max]*len(joints))

# 
#ocp.constraints.idxsbh = np.arange(nx)

# cost for soft constraints
#ocp.cost.Zu = 10*np.eye(len(ocp.constraints.idxsbu))
#ocp.cost.Zl = 10*np.eye(len(ocp.constraints.idxsbu))

ocp.constraints.lbu = np.array([-ddq_max]*nu)
ocp.constraints.ubu = np.array([ddq_max]*nu)
ocp.constraints.idxbu = np.arange(nu)

# Initial state constraint. Populated at run-time.
ocp.constraints.lbx_0 = np.zeros(nx)
ocp.constraints.ubx_0 = np.zeros(nx)
ocp.constraints.idxbx_0 = np.arange(nx)


# Terminal state constraint (same logic as path constraints).
#model.con_h_expr_e = model.con_h_expr
#ocp.constraints.lh_e = ocp.constraints.lh
#ocp.constraints.uh_e = ocp.constraints.uh
#ocp.constraints.idxsbh_e = ocp.constraints.idxsbh

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

execdir = os.getcwd()

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
state_names = [
  joint.name() + '/position' for joint in joints
] + [
  joint.name() + '/velocity' for joint in joints
]

 
get_input_names = create_metadata_getter(
  'get_input_names', 
  {
    'states': state_names,
    # We have technically constrained the entire state, but via h(x). The state isn't constrained directly.
    'path_state_constraints': [],
    'terminal_state_constraints': [],
    'initial_state_constraints': state_names,
    'path_references': path_reference_names,
    'terminal_references': terminal_reference_names,
    'controls': [f'{joint.name()}/acceleration' for joint in joints_actuated],
  }
)

get_acados_prefix = f'''
const char* get_acados_prefix() {{
  return "{model.name}"; 
}}
'''

metadata_code = create_metadata_file([get_input_names, get_acados_prefix])

with open(os.path.join(code_dir, 'controller_metadata.cpp'), 'w') as f:

  f.write(metadata_code)

builder.exec(code_dir)

if len(sys.argv) >= 3:
  lib_install_location = sys.argv[2]
else:
  lib_install_location = execdir

os.system(f'mv {code_dir}/libacados_ocp_solver_{model.name}.so {lib_install_location}')
