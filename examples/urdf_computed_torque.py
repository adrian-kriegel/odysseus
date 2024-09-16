#!/usr/bin/env python

'''
Loads a URDF file and creates a shared library for computing the dynamics of the robot.

This library can then be loaded by a controller to implement efficient exact linearization via state feedback.

You can pipe a URDF into this script followed by the name of the base link of the robot and the path to the shared library to generate.

For example:

xacro rrbot.xacro | ./urdf_computed_torque.py link1 ./my_library.so

or 

cat rrbot.urdf | ./urdf_computed_torque.py link1 ./my_library.so

'''

import sys

import os

import tempfile

import xml.dom.minidom as xml

import casadi as ca

from symengine import Matrix, Symbol

from odysseus import diff, URDFModel
from odysseus.symengine2casadi import to_casadi

# Read urdf from stdin (this just makes it easier to pipe output from xacro into this script).
doc = xml.parseString(sys.stdin.read())

# Read base link name from command args.
base_link_name = sys.argv[1]

# Create a model from the URDF.
urdf = URDFModel(doc)

def get_robot_name():
  return urdf.dom_.getElementsByTagName('robot')[0].getAttribute('name')

# Where to write the shared library to.
lib_file = len(sys.argv) > 2 and sys.argv[2] or os.path.abspath(f'./{get_robot_name()}_ct_library.so')


# Create a link between the world and the element we want to load from the URDF.
world_to_base = urdf.model().create_root(
  name='world_to_base',
  # As in Example 1, using 
  # functions of 't' would allow the robot to move, 
  # adding generalized coordinates. 
  # In this case, the robot's base is fixed. 
  xyz=Matrix([0,0,0]),
  rpy=Matrix([0,0,0])
)

# Find and load the element named `base_link_name` from the URDF.
# This will also load all child elements.
urdf.add_element_by_name(
  # Find the element named `base_link_name` in the URDF.
  base_link_name,
  # Add this link as a child of world_to_base.
  world_to_base
)

model = urdf.model()

# TODO: Read actuated/free joint names from command line args.
joints_actuated = [model.find_joint('joint1'), model.find_joint('joint2')]
joints_free = []

joints = joints_free + joints_actuated

print('Joints: ', [joint.name_ for joint in joints])

q = model.q(joints)
dq = diff(q, 't')
ddq = diff(dq, 't')
n = len(q)

# number of actuated joints
n_act = len(joints_actuated)
# number of non-actuated joints
n_free = len(joints_free)

print('n = ', n)

print('Deriving dynamics...')

# Note, the shape is: mass*ddq + tau = u
# Sometimes, you will find mass*ddq = tau + u, so be cautious to flip the sign of tau.
mass,tau = model.canonical_dynamics()

# CasADi has a mucher nicer codegen interface than SymEngine/SymPy. 
# We therefore transfer all equations to CasADi.

print('Converting to CasADi...')

ca_q = ca.MX.sym('q', n)
ca_dq = ca.MX.sym('dq', n)

# SymEngine to CasADi substitutions.
ca_subs = {
  x: ca_x for x, ca_x in 
  list(zip(dq, ca.vertsplit(ca_dq, 1))) + 
  list(zip(q, ca.vertsplit(ca_q, 1)))
} | { Symbol('g'): 9.81 }

ca_mass = to_casadi(mass, ca_subs)
ca_tau = to_casadi(tau, ca_subs)

funcname = 'dynamics'

# CasADi allows for multiple inputs but this results the caller to un-pack the inputs and re-pack them into a dict.
# So we just pack them into one vector before generating the code, which allows the caller to also just pack all inputs into one vector.
dynamics_inputs = ca.vertcat(ca_q, ca_dq)

# TODO: It would be more efficient to compute the upper triangle of the mass matrix
# as it is symmetric. 
dynamics_func = ca.Function(funcname, [dynamics_inputs], [ca_mass, ca_tau])

# Map joint ROS2 control state interfaces to the system state [q, dq].
# These are the input to our controller.
state_interfaces = [
  joint.name() + '/position' for joint in joints
] + [
  joint.name() + '/velocity' for joint in joints
]

# Map joint ROS2 control command interfaces to the system input.
# These are the outputs of our controller.
command_interfaces = [
  joint.name() + '/effort' for joint in joints_actuated
]

print('Generating code...')

# Create tmp dir containing the source.

tempdir = tempfile.mkdtemp()

os.chdir(tempdir)

cfile = dynamics_func.generate()

print('Generated source file: ', os.path.join(tempdir, cfile))

# Add metadata to the library.
def to_cpp_initializer_list(l):

  return ', '.join([f'"{x}"' for x in l])

metadata_code = f'''
#include <string>
#include <vector>

extern "C" {{

void get_controller_metadata(
  std::vector<std::string>& joints,
  std::vector<std::string>& joints_actuated,
  std::vector<std::string>& joints_free,
  std::vector<std::string>& state_interfaces,
  std::vector<std::string>& command_interfaces
){{
  joints = {{
    {to_cpp_initializer_list([joint.name() for joint in joints])}
  }};
  joints_actuated = {{
    {to_cpp_initializer_list([joint.name() for joint in joints_actuated])}
  }};
  joints_free = {{
    {to_cpp_initializer_list([joint.name() for joint in joints_free])}  
  }};
  state_interfaces = {{
    {to_cpp_initializer_list(state_interfaces)}
  }};
  command_interfaces = {{
    {to_cpp_initializer_list(command_interfaces)} 
  }};
}}

}}
'''

with open(os.path.join(tempdir, 'metadata.cpp'), 'w') as f:
  f.write(metadata_code)

print('Compiling...')
assert os.system(f'gcc -Ofast -c -fPIC {cfile} -o {funcname}.o') == 0
assert os.system(f'g++ -c -fPIC metadata.cpp -o metadata.o') == 0
assert os.system(f'gcc -shared -o {lib_file} {funcname}.o metadata.o') == 0

print('Generated shared library: ', lib_file)

