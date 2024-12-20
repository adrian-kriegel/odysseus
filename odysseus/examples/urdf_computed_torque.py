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

import subprocess

import tempfile

import casadi as ca

from symengine import Matrix, Symbol

from odysseus import diff
from odysseus.symengine2casadi import SymEngineToCasADiConverter

from odysseus.examples.model_from_args import urdf, base_link_name, get_robot_name
from odysseus.examples.inject_metadata import create_metadata_getter, create_metadata_file

def main():

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

  to_casadi = SymEngineToCasADiConverter([
    (dq, ca_dq),
    (q, ca_q),
    (Symbol('g'), 9.81)
  ])

  ca_mass = to_casadi(mass)
  ca_tau = to_casadi(tau)

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

  # Prepare metadata to inject into the generated library.
  get_controller_metadata = create_metadata_getter(
    'get_controller_metadata', 
    {
      'joints': [joint.name() for joint in joints],
      'joints_actuated': [joint.name() for joint in joints_actuated],
      'joints_free': [joint.name() for joint in joints_free],
      'state_interfaces': state_interfaces,
      'command_interfaces': command_interfaces,
    }
  )

  metadata_code = create_metadata_file([get_controller_metadata])

  with open(os.path.join(tempdir, 'metadata.cpp'), 'w') as f:
    f.write(metadata_code)

  print('Compiling...')

  def run_command(cmd):
    print(cmd)
    result = subprocess.run(cmd.split(' '), capture_output=True)
    result.check_returncode()

  run_command(f'gcc -Ofast -c -fPIC {cfile} -o {funcname}.o')
  run_command(f'g++ -c -fPIC metadata.cpp -o metadata.o')
  run_command(f'g++ -shared -o {lib_file} {funcname}.o metadata.o -lstdc++')

  print('Generated shared library: ', lib_file)

