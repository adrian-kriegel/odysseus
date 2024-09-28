'''
Helper used by examples in order to load a model from URDF from stdin.
'''

import sys

import xml.dom.minidom as xml

from odysseus import URDFModel

# Read urdf from stdin (this just makes it easier to pipe output from xacro into this script).
doc = xml.parseString(sys.stdin.read())

# Read base link name from command args.
base_link_name = sys.argv[1]

# Create a model from the URDF.
urdf = URDFModel(doc)

def get_robot_name():
  return urdf.dom_.getElementsByTagName('robot')[0].getAttribute('name')
