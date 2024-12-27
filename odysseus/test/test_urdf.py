
import pathlib

import xml.dom.minidom as xml

from symengine import Matrix, cos, sin, pi, eye

from odysseus import URDFModel, URDFElement, Link, Transform

def test_urdf_transforms():
    '''
    Loads a more complex model than the rrbot (ANYmal D) and checks some transformations that I have calculated by hand.
    '''    
    doc = xml.parse(pathlib.Path(__file__).parent.joinpath('res/anymal_d_simple_description/urdf/anymal.urdf').__str__())

    robot = URDFModel(doc)

    assert robot.get_robot_name() == 'anymal'

    joints_external = [
        'inspection_payload_pan_to_tilt',
        'inspection_payload_mount_to_pan'
    ]

    def make_external_fixed(element : URDFElement, parent : Link):
        '''
        Loads any joints from joints_external as Link instead of JointRevolute.
        '''

        name =  element.attr('name')

        if name in joints_external:
            return Link(name, parent, element.origin())

    robot.transform_elements(make_external_fixed)

    model = robot.model()

    world_to_base = model.create_root(
        'world_to_base',
        Matrix([0,0,0]),
        Matrix([0,0,0])
    )

    robot.add_element_by_name(
        'base',
        world_to_base
    )

    # transform LF_KFE <-> LF_KFE_drive

    kfe = model.find('LF_KFE')

    kfe_transform = kfe.get_transform('LF_KFE_drive')

    assert kfe_transform.rot_ == Matrix([
        [1, 0, 0],
        [0, cos(kfe.q()), -sin(kfe.q())],
        [0, sin(kfe.q()), cos(kfe.q())]
    ])

    assert kfe_transform.trans_ == Matrix([0,0,0])

    # transform LF_SHANK <-> LF_KFE

    shank = model.find('LF_SHANK')

    shank_transform = shank.get_transform('LF_KFE')

    assert shank_transform.rot_ == eye(3)

    assert shank_transform.trans_ == Matrix([0,0,0])

    # LF_shank_fixed  <-> LF_SHANK

    shank_fixed = model.find('LF_shank_fixed')

    shank_fixed_transform = shank_fixed.get_transform('LF_SHANK')

    assert shank_fixed_transform.rot_ == Matrix([
        [cos(-pi/2), -sin(-pi/2), 0],
        [sin(-pi/2), cos(-pi/2), 0],
        [0, 0, 1]
    ])

    assert shank_fixed_transform.trans_ == Matrix([0,0,0])

    # LF_FOOT  <-> LF_shank_fixed

    foot = model.find('LF_FOOT')

    foot_transform = foot.get_transform('LF_shank_fixed')

    assert foot_transform.rot_ == eye(3)

    assert foot_transform.trans_ == Matrix([0.1, 0.02225, -0.39246])

    # complete transform chain

    transform = foot.get_transform('LF_KFE_drive')

    assert transform == kfe_transform * shank_fixed_transform * foot_transform