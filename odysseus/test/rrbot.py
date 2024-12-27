
from symengine import Matrix

from odysseus import Segment, Joint, Link, JointRevolute, Transform, inertia_matrix, LinkModel, diff


def model_rrbot(
    xyz = Matrix([0,0,0]),
    rpy = Matrix([0,0,0]),
    l0 = 1, 
    l1 = 1.1, 
    m0 = 1, 
    m1 = 1.1, 
    j0 = 0.5, 
    j1 = 0.6,
    d0 = 0.7,
    d1 = 0.7
):
    '''
    Returns LinkModel of the rrbot from the Gazebo examples with different defaults.

    x, y, z     -- Root coordinates.

    '''

    J0 = inertia_matrix(iyy=0.06)
    J1 = inertia_matrix(iyy=0.08)

    model = LinkModel()

    root = model.create_root('root', xyz, rpy)

    joint0 = JointRevolute(
        'joint0',
        root,
        origin=Transform(),
        axis=Matrix([0,1,0]),
        damping=d0
    )

    segment0 = Segment(
        'segment0', 
        joint0, 
        Transform(), 
        m0, 
        J0, 
        inertial_origin=Matrix([0,0,l0/2])
    )

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
        inertial_origin=Matrix([0,0,l1/2])
    )

    return model

