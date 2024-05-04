
'''
Tests for the API provided by LinkModel, Link, Segment etc.
Correctness of the symbolic terms, dynamics etc. is not subject to this suite.
'''

from sympy import Matrix, symbols

from odysseus import inertia_matrix, Segment, JointRevolute, LinkModel, Transform

def test_link_model():

    model = LinkModel()

    root = model.create_root('root', symbols('x y z'), symbols('rx ry rz'))

    joint0 = JointRevolute(
        'joint0',
        root,
        origin=Transform(),
        axis=Matrix([0,1,0]),
        damping=0.7
    )

    segment0 = Segment('segment0', joint0, Transform(), 1, inertia_matrix(), inertial_offset=Matrix([0,0,0.5]))

    joint1 = JointRevolute(
        'joint1',
        segment0,
        origin=Transform(Matrix([0, 0, 1])),
        axis=Matrix([0,1,0]),
        damping=0.7
    )
    
    assert model.find('root') == root
    assert model.find('joint0') == joint0
    assert model.find('segment0') == segment0
    assert model.find('joint1') == joint1
    assert model.find('joint2') == None

    assert model.find_joint('joint0') == joint0
    assert model.find_segment('segment0') == segment0
    assert model.find_joint('joint1') == joint1

    assert model.find_segment('joint0') == None
    assert model.find_joint('segment0') == None
    assert model.find_segment('joint1') == None