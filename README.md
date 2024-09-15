
# ODYSSEUS

Odysseus (**O**ptimal **DY**namic **S**ystems using **S**ymbolic **EU**ler-Lagrange equation**S** - yes, really) is a Python library for symbolically deriving the dynamics equations that govern the motion of mechatronic systems. The library allows for the creation of simulations, computed torque controllers, or NMPC controllers from robot descriptions such as URDF.

## Example 0: Efficient controllers from a URDF 

The script `examples/urdf_computed_torque.py` generates a shared library that can be loaded by a controller to realize efficient exact linearization via state feedback.

## Example 1: Manual Construction

The following pieces of code are taken from `examples/rrbot_model.py`. The snippets are incomplete. Consult the source for a runnable example. The example shows how to manually construct a joint-actuated inverted pendulum. Example 2 shows how to load a model from a URDF file. Class descriptions and more detailed explanations can be found in the respective docstrings.

The first step is to create an empty model and add a root element. In URDF, this is often called the **base_link**. In the case of the robotic arm, this is fixed to the world. 

```Python
model = LinkModel()

root = model.create_root(
    name='root', 
    xyz=Matrix([0,0,0]), 
    ryp=Matrix([0,0,0])
)
```

Or if you want your robot to be able to move around, replace any or all coordinates with time dependent variables.

```Python
from symengine import Function

x = Function('x')('t')

root = model.create_root(
    name='root', 
    xyz=Matrix([x,0,0]), 
    ryp=Matrix([0,0,0])
)
```

We now add a revolute joint to the root:

```Python
joint0 = JointRevolute(
    'joint0',
    parent=root,
    origin=Transform(),
    axis=Matrix([0,1,0]),
    damping=d0
)
```

The joint has a generalized coordinate:

```Python
print(joint0.q()) # --> q_joint0(t)
```

A `Segment` is a Link that contributes to the systems total energy, meaning it has nonzero mass or inertia. Note that any variables, such as mass, inertia, lengths, etc. may be symbolic.

```Python
segment0 = Segment(
    'segment0', 
    parent=joint0, 
    # Identity Transform 
    # (use the same frame as joint0)
    Transform(), 
    mass=m0, 
    inertia=J0, 
    inertial_offset=Matrix([0,0,l0/2])
)
```

Repeat the same steps to add the second joint and segment:

```Python
joint1 = JointRevolute(
    'joint1',
    segment0,
    origin=Transform(Matrix([0, 0, l0])),
    axis=Matrix([0,1,0]),
    damping=d1
)

segment1 = Segment(
    'segment1', 
    joint1, 
    Transform(), 
    m1, 
    J1, 
    inertial_offset=Matrix([0,0,l1/2])
)
```

Generate symengine expressions describing the dynamics of the system: 

```Python
# vector of generalized coordinates
q = model.q()

# get dynamics for all joints:
dynamics = model.dynamics()

# get dynamics for specific joints
# in a specific order
dynamics = model.dynamics([joint0, joint1])

# get dynamics as M(q,dq)*ddq + rest(q,dq) = external_forces
mass, rest = model.canonical_dynamics()
```

### Example 2: Loading a URDF

It is not feasible to model complicated assemblies manually as in the first example. While not all features are supported, most URDF files can be imported into Odysseus. You can also load only parts of URDF files. This allows you to, for example, model just the dynamics of one leg.

```Python
import xml.dom.minidom as xml

doc = xml.parse('path/to/robot.urdf')

urdf = URDFModel(doc)

# create a link between the world and 
# the element we want to load from the URDF
world_to_base = urdf.model().create_root(
    name='world_to_base',
    # As in Example 1, using 
    # functions of 't' would allow the robot to move, 
    # adding generalized coordinates. 
    # In this case, the robot is fixed. 
    xyz=Matrix([0,0,0]),
    rot=Matrix([0,0,0])
)

# Load the element named 'base' from the URDF.
# This will also load all child elements.
urdf.add_element_by_name(
    'base',
    world_to_base
)

# Use the model ...
dynamics = urdf.model().dynamics()
```
