
from symengine import Matrix, symbols, eye, sin, cos, Symbol
import sympy as sp

from odysseus import Segment, Joint, Link, JointRevolute, Transform, inertia_matrix, LinkModel, diff

from odysseus.test.rrbot import model_rrbot

from odysseus.test.util import assert_eq

def test_rrbot():

    '''
    Asserts correctness of the model generated using sym_lm by comparing the terms to manually derived terms.
    The robot we are modelling here is similar to the rrbot from the Gazebo examples.
    The rrbot's original model has the two arms share the same parameters.
    Therefore parameters are deliberately different from those of the rrbot to ensure that nothing is mixed up. 
    '''

    # axle_offset is the pivot point relative to the end of each rod
    axle_offset = 0.05

    # we don't care about the actual length of the segments, but the lenths between the axles
    l0 = 1 - axle_offset*2
    l1 = 1.1 - axle_offset*2

    # mass
    m0 = 1
    m1 = 1.5

    # damping
    d0 = 0.7
    d1 = 0.8

    # moment of inertia
    j0 = 0.06
    j1 = 0.08

    # root coordinates 
    x,y,z = symbols('x y z')

    model = model_rrbot(Matrix([x,y,z]), Matrix([0,0,0]),l0,l1,m0,m1,j0,j1,d0,d1)

    segment0, segment1 = model.segments()
    root, joint0, joint1 = model.joints()

    ## check that the model matches the manually determined equations

    x0,y0,z0 = segment0.global_inertial_origin()
    x1,y1,z1 = segment1.global_inertial_origin()

    transform = root.get_transform('root')
    transform1 = segment0.get_transform('root')
    transform2 = segment1.get_transform('root')

    assert transform.trans_ == Matrix([0,0,0])

    assert transform2 == transform1 * joint1.origin_ * segment0.origin_

    # the robot rotates around the y coordinate, so it should remain the same
    assert y0 == y
    assert y1 == y

    assert x0 == x + sin(joint0.q_) * l0/2
    assert z0 == z + cos(joint0.q_) * l0/2

    # TODO: assert_eq does not make use of added angle identity:
    # sin(a+b) = sin(a)*cos(b) + sin(b)*cos(a)
    # rhs could be written more compactly: 
    # x + sin(joint0.q_) * l0 + sin(joint0.q_ + joint1.q_) * l1/2
    assert_eq(x1, x + 0.5*(sin(joint0.q())*cos(joint1.q()) + sin(joint1.q())*cos(joint0.q())) + 0.9*sin(joint0.q()))
    assert_eq(z1, z + 0.5*(-sin(joint0.q())*sin(joint1.q()) + cos(joint0.q())*cos(joint1.q())) + 0.9*cos(joint0.q()))

    assert_eq(segment0.v_rot(), Matrix([0, diff(joint0.q_, 't'), 0]))
    assert_eq(segment1.v_rot(), Matrix([0, diff(joint0.q_ + joint1.q_, 't'), 0]))

    # assert transform1.angles() == Matrix([0, joint0.q_, 0])

    #
    # check energy terms
    #

    t = Symbol('t')

    # linear kinetic energy
    assert_eq(segment0.lin_energy(),0.5 * m0 * (diff(x0, t)**2 + diff(z0, t)**2 + diff(y, t)**2))
    assert_eq(segment1.lin_energy(),0.5 * m1 * (diff(x1, t)**2 + diff(z1, t)**2 + diff(y, t)**2))

    # rotational kinetic energy
    dq0 = diff(joint0.q_, t)
    dq1 = diff(joint1.q_, t)

    assert_eq(sp.simplify(segment0.rot_energy()), sp.simplify(0.5 * j0 * dq0**2))
    assert_eq(sp.simplify(segment1.rot_energy()), sp.simplify(0.5 * j1 * (dq0 + dq1)**2))
    
    # potential energy

    g = Symbol('g')

    assert_eq(sp.simplify(segment0.pot_energy()), m0 * g * z0)
    assert_eq(sp.simplify(segment1.pot_energy()), m1 * g * z1)

    #
    # Assert the resulting dynamics.
    # 

    # we have checked that the energies are correct
    T = segment0.lin_energy() + segment0.rot_energy() + segment1.lin_energy() + segment1.rot_energy()
    U = segment0.pot_energy() + segment1.pot_energy()
    
    t = Symbol('t')

    L = T - U

    assert_eq(model.lagrangian(), L)

    damping_torques = Matrix([-d0*dq0, -d1*dq1])

    q = Matrix([joint0.q(), joint1.q()])
    dq = diff(q, t)
    ddq = diff(dq, t)

    assert_eq(model.q(), q)

    tau_expected = Matrix([
        diff(diff(L, dq[0]), t) - diff(L, q[0]) - damping_torques[0],
        diff(diff(L, dq[1]), t) - diff(L, q[1]) - damping_torques[1]
    ])

    # assert the "unshaped" dynamics generated for each coordinate 
    dynamics = model.generate_dynamics()

    assert_eq(next(dynamics), tau_expected[0])
    assert_eq(next(dynamics), tau_expected[1])

    # assert the "canonical" form which is m(q,dq)ddq + g(q,dq)
    mass, rest = model.canonical_dynamics()

    # rest must not depend on ddq 
    assert diff(rest, ddq) == Matrix([[0,0], [0,0]])

    # putting the terms back together yields the same dynamics
    assert_eq(mass*ddq + rest, tau_expected)