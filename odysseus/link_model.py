#!/bin/env python

from __future__ import annotations 

from enum import Enum
from typing import Self

import numpy as np

from symengine import Symbol, symbols, Matrix, cos, sin, Function, eye, atan2, sqrt
import symengine as se

from odysseus.sym_util import approximate_integers


def diff(f, x, foreach=False):
    '''
    Unlike sympy, symengine does not support diff(f, x) if x is a vector (Matrix).
    diff_by_vec patches this missing feature.

    foreach   -- Will return partial derivatives in each row if set to True. Will return Jacobian otherwise.
    '''
    if isinstance(x, Matrix):
        if isinstance(f, Matrix):
            res = Matrix([
                Matrix(se.diff(f, v)).transpose() for v in x
            ])
        else:
            res = Matrix([
                se.diff(f, v) for v in x
            ])

        if foreach:
            return res
        else:
            return res.transpose()
    else:
        return se.diff(f, x)


class Transform:
    '''
    Symbolic 3D rigid transformation.
    '''
    def __init__(self, trans : Matrix | None = None, angles : Matrix | None = None, rot : Matrix | None = None):

        self.trans_ = Matrix([0,0,0]) if trans is None else trans
        
        if angles is not None and rot is not None:
            raise Exception('Construct Transform either using angles or rot.')
        
        if angles is not None:
            self.rot_ = Transform._rpy_to_matrix(angles)
        elif rot is not None:
            self.rot_ = rot
        else:
            self.rot_ = eye(3) 

    def inverse(self) -> Transform:
        rt = self.rot_.transpose()
        return Transform(
            - rt*self.trans_,
            rot=rt
        )

    def chain(self, t : Self):
        '''
        Returns self * t.
        '''
        return Transform(
            self.apply_to(t.trans_),
            rot=self.rot_*t.rot_
        )

    def apply_to(self, p : Matrix):
        return self.rot_*p + self.trans_

    def __mul__(self, t : Self | Matrix):

        if isinstance(t, Matrix):
            return self.apply_to(t)
        elif isinstance(t, Transform):
            return self.chain(t)
        else:
            raise Exception(f"Cannot __mul__ Transform and {type(t)} {t}")

    def __eq__(self, t : Self):

        return (
            self.trans_ == t.trans_ and
            self.rot_ == self.rot_
        )

    @staticmethod
    def from_axis_angle(axis : Matrix, angle):

        return Transform(Matrix([0, 0, 0]), axis*angle)

    @staticmethod
    def _rpy_to_matrix(rpy) -> Matrix:
        '''
        Convert RPY to a rotation matrix.
        '''
        roll, pitch, yaw = rpy
        cr = cos(roll)
        sr = sin(roll)
        cp = cos(pitch)
        sp = sin(pitch)
        cy = cos(yaw)
        sy = sin(yaw)

        return Matrix([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0,0,1]
        ]) * Matrix([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp,0,cp]
        ]) * Matrix([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])

class Link:

    ''' 
    Represents any part of a Joint/Segment assembly.
    Each Link is attached to a parent Link (except for the root Link).
    Each Link defines its own coordinate system via self.origin_. This is the transformation from the parent's coordinate system to the Links own coordinate system.
    '''

    def __init__(
        self,
        name : str,
        parent : Self | None,
        origin : Transform,
        model : LinkModel | None = None,
        fixed = True
    ):
        self.name_ = name
        self.origin_ = origin
        self.parent_ = parent
        self.fixed_ = fixed
        
        if self.parent_ is None and model is None:
            raise Exception('Every Link must have either a parent or be a root Link in a LinkModel.')

        self.model_ = model if self.parent_ is None else self.parent_.model_

        self.model_.register(self)

        self.children_ : list[Link] = []

        # caching the world transform as it is often needed
        # I don't want to compute it in the constructor though, as the user should 
        # be allowed to change the structure of the model.
        self.world_transform_ : Transform | None = None

        if self.parent_ is not None:
            self.parent_.register_child(self)

    def register_child(self, child : Link):
        self.children_.append(child)

    def get_transform(self, to_name : str | None = None, from_name = None) -> Transform:
        '''
        Get the Transform from the node named from_name to to_name (=self.name_).
        If no to_name is specified, the transform from self to world will be returned.
        '''

        if to_name is None and self.world_transform_ is not None:
            return self.world_transform_

        # from_name is mainly for printing error messages
        if from_name is None:
            from_name = self.name_
        
        if from_name == to_name or to_name == self.name_:
            res = Transform()

        elif self.parent_ is not None:
            res = self.parent_.get_transform(to_name, from_name) * self.origin_
        elif to_name == None:
            res = self.origin_
        else:
            raise Exception(f'Could not find chain from {from_name} to {to_name}')

        if to_name is None:
            self.world_transform_ = res
        
        return res

    def v_rot(self):
        '''
        Angular velocity.
        '''
        r = self.get_transform().rot_

        omega = diff(r, 't') * r.transpose()

        return Matrix([
            omega[2,1],
            omega[0,2],
            omega[1,0]
        ])

class ActuationType(Enum):
    '''
    Describes how/if a joint might be actuated.
    '''

    # Not (directly) actuated: The joint moves entirely based on the dynamics.
    FREE = 1

    # Not (directly or indirectly) actuated: 
    # This is equivalent to a parametrized fixed joint in that the dof is governed by some external influence.
    # This can be used to model joints that have their own controller attached.
    EXTERNAL = 2
     
    # Directly actuated joint.
    DIRECT = 3


class Joint(Link):
    '''
    Abstract base class for links with degrees of freedom. 
    A 'fixed' joint (as in URDF) is represented using the Link class.
    '''
    def __init__(
        self,
        name : str,
        origin : Transform,
        parent : Link,
        actuation : ActuationType = ActuationType.DIRECT,
        model : LinkModel | None = None
    ):
        Link.__init__(self, name, origin, parent, fixed=False, model=model)

        t = Symbol('t')

        # generalized coordinate representing the dof
        self.q_ = Function(f'q_{name}')(t)

        # actuation type of this joint
        self.actuation_ = actuation

        self.dof_ = 1

    def q(self):
        ''' 
        Getter for the generalized coordinate.
        '''
        return self.q_
    
    def dof(self):
        '''
        Returns number of dof.
        '''
        return len(self.q_) if isinstance(self.q_, Matrix) else 1

    def damping_force(self):
        '''
        Getter for the generalized damping force.
        '''
        raise NotImplemented()

class JointLimits:

    def __init__(self, pos_l = None, pos_u = None, vel_l = None, vel_u = None, idx_q : list[int] = []):

        self.pos_l_ = pos_l
        self.pos_u_ = pos_u
        self.vel_l_ = vel_l
        self.vel_u_ = vel_u

        # Indices of q that are affected by the limits
        self.idx_q_ = []

class JointFree(Joint):
    '''
    Joint moving freely according to its origins DOF.
    This is the only joint that is not required to be attached to a parent link.
    '''
    def __init__(
        self,
        name : str,
        parent,
        xyz : Matrix,
        rpy : Matrix,
        q : Matrix,
        actuation : ActuationType = ActuationType.DIRECT,
        model : LinkModel | None = None,
        limits : JointLimits | None = None
    ):
        '''
        Keyword arguments:
        name        -- Name.
        xyz         -- Transform xyz, may contain terms from q.
        rpy         -- Transform rpy, may contain terms from q.
        q           -- Degrees of freedom in xyz/rpy.
        model       -- LinkModel.
        '''

        self.limits_ = limits

        origin = Transform(xyz, rpy)

        Joint.__init__(self, name, parent, origin, actuation=actuation, model=model)

        if q.rows > 0 and diff(q, 't').is_zero_matrix:
            raise Exception('q must be a function of time.')

        self.q_ = q

        self.dof_ = len(q)

    def damping_force(self):
        return Matrix([ 0 for q in self.q() ])


class JointRevolute(Joint):

    def __init__(
        self,
        name : str,
        parent : Link,
        origin : Transform,
        axis : Matrix,
        damping = 0,
        actuation : ActuationType = ActuationType.DIRECT,
        limits : JointLimits = JointLimits()
    ):
        '''
        Keyword arguments:
        name        -- Name.
        parent      -- Parent link.
        origin      -- Origin transformation from the parent link.
        axis        -- Axis of rotation (normalized vector) 
        damping     -- Scalar damping factor.
        '''

        Joint.__init__(self, name, parent, origin, actuation=actuation)

        self.axis_ = axis

        # chain the dynamic part of the origin to the fixed origin TODO: I don't like this :(
        self.origin_ = self.origin_ * Transform.from_axis_angle(self.axis_, self.q())

        self.damping_ = damping

        self.limits_ = limits

    def damping_torque(self):
        '''
        Returns torque induced via damping.
        '''
        return se.diff(self.q(), 't') * self.damping_

    def damping_force(self):
        return self.damping_torque()

class JointLinear(Joint):

    def __init__(
        self,
        name : str,
        parent : Link,
        origin : Transform,
        axis : Matrix,
        damping = 0,
        actuation : ActuationType = ActuationType.DIRECT
    ):
        '''
        Keyword arguments:
        name        -- Name.
        parent      -- Parent link.
        origin      -- Origin transformation from the parent link.
        axis        -- Axis of actuation (normalized vector)
        damping     -- Scalar damping factor.
        '''

        Joint.__init__(self, name, parent, origin, actuation=actuation)

        self.axis_ = axis

        # chain the dynamic part of the origin to the fixed origin TODO: I don't like this :(
        self.origin_ = self.origin_ * Transform(self.axis_ * self.q())

        self.damping_ = damping

    def damping_force(self):
        return se.diff(self.q(), 't') * self.damping_


class Segment(Link):

    '''
    Represents a link with inertial properties.
    Energies are w.r.t. the intertial point (global inertial origin).
    '''

    def __init__(
        self,
        name : str,
        parent : Link,
        origin : Transform,
        mass,
        inertia : Matrix,
        inertial_offset = None
    ):
        '''
        Keyword arguments:
        name            -- Name.
        parent          -- Parent Link or Joint
        origin          -- Transformation from the parent joint to the segment.
        mass            -- Mass term (symbolic or float). 
        inertia         -- Inertia matrix.
        inertial_offset -- Offset between origin and the inertial origin.
        '''
        Link.__init__(self, name, parent=parent, origin=origin)

        self.m_ = mass   
        self.inertia_ = inertia
        self.inertial_origin_ = Matrix([0,0,0]) if inertial_offset is None else inertial_offset

    def global_inertial_origin(self):
        ''' Returns the global position of the inertial origin. '''
        return self.get_transform() * self.inertial_origin_


    def lin_energy(self):

        '''
        Linear kinetic energy.
        '''

        x,y,z = self.global_inertial_origin()

        t = Symbol('t')

        return 0.5 * self.m_ * (diff(x,t)**2 + diff(y,t)**2 + diff(z,t)**2)

    def rot_energy(self):
        '''
        Rotational kinetic energy.
        '''

        v_rot = self.v_rot()
        # the expression in the parantheses yields a Matrix of shape 1x1 but we want a scalar
        return (0.5 * v_rot.transpose() * self.inertia_ * v_rot)[0]

    def kin_energy(self):
        ''' 
        Total kinetic energy.
        '''

        return self.lin_energy() + self.rot_energy()

    def pot_energy(self, g=Symbol('g')):
        '''
        Potential (gravitational) energy.
        '''
        
        _,_,z = self.global_inertial_origin()

        return self.m_ * g * z

class LinkModel:

    '''
    Acts as a type of arena/collection for Links.

    Limitations:
    
        - The model must have a tree-like structure. Circular links are not supported.
        - There may only be one segment attached to one joint. However, there may be multiple joints per segment.
    '''

    def __init__(self):

        # all elements in the model
        self.links_ : list[Link] = []
        
        # all elements with degrees of freedom
        self.joints_ : list[Joint] = []

        # all elements with inertia/mass
        self.segments_ : list[Segment] = []

    def links(self):
        return self.links_

    def joints(self):
        return self.joints_
    
    def segments(self):
        return self.segments_

    def partition(self, predicate):
        '''
        Partition the joints in the system according to a predicate.
        Re-orders joints into [a...,b...] such that the predicate is true for joints in a.
        '''
        a = []
        b = []

        for joint in self.joints_:
            if predicate(joint):
                a.append(joint)
            else:
                b.append(joint)
        
        self.joints_ = a + b

    def create_root(
        self,
        name : str = 'root',
        xyz = Matrix(symbols('x y z')),
        rpy = Matrix(symbols('rx ry rz')),
        q : Matrix | None = None,
        actuation : ActuationType = ActuationType.FREE
    ) -> Self:
        '''
        Creates a root link. Any (great-grand) child added to this link will be added to the model.
        '''

        if q == None:
            
            q = Matrix([ v for v in (list(xyz) + list(rpy)) if diff(v, 't') != 0 ])
            
        return JointFree(name, None, xyz, rpy, q, actuation, model=self)

    def register(self, link : Link):

        self.links_.append(link)

        if isinstance(link, Joint):
            self.joints_.append(link)
        elif isinstance(link, Segment):
            self.segments_.append(link)

    def find(self, name : str) -> Link | None:
        ''' 
        Returns Link by name or None if not found. 
        '''
        return next((link for link in self.links_ if link.name_ == name), None)

    def find_segment(self, name : str) -> Segment | None:
        ''' 
        Returns Segment by name or None if not found. 
        '''
        return next((link for link in self.segments_ if link.name_ == name), None)
    
    def find_joint(self, name : str) -> Segment | None:
        ''' 
        Returns Joint by name or None if not found. 
        '''
        return next((link for link in self.joints_ if link.name_ == name), None)

    def kin_energy(self):
        '''
        Returns the kinetic energy of the entire system.
        '''

        return sum([segment.kin_energy() for segment in self.segments_], se.Float(0))

    def pot_energy(self):
        '''
        Returns the potential energy of the entire system.
        '''

        return sum([segment.pot_energy() for segment in self.segments_], se.Float(0))

    def lagrangian(self):
        '''
        Returns the lagrangian of the entire system.
        '''

        return self.kin_energy() - self.pot_energy()

    def generate_dynamics(self, joints : list[Joint] | None = None):
        '''
        Generator for the dynamics f(q,dq,ddq) such that f(q,dq,ddq) = external_forces for each single coordinate.
        '''

        if joints is None:
            joints = self.joints_

        t = Symbol('t')
        
        L = self.lagrangian()

        # helper for dynamics per coordinate
        def dynamics_helper(q, damping_forces):
            return diff(diff(L, diff(q, t), foreach=True), t) - diff(L, q, foreach=True) - damping_forces

        for joint in joints:
            q = joint.q()
            f = joint.damping_force()

            if isinstance(q, Matrix):
                yield from (dynamics_helper(qi, fi) for qi,fi in zip(q, f))
            else:
                yield dynamics_helper(q, f)

    def q(self, joints : list[Joint] | None = None):
        '''
        Returns vector (Matrix) of generalized coordinates.
        '''
        if joints is None:
            joints = self.joints()

        q = []

        # collect all coordinates 
        # TODO: joint.q() should be the same type for all joints! 
        for joint in joints:
            jq = joint.q()
            if isinstance(jq, Matrix):
                q = q + list(jq)
            else:
                q.append(jq)

        return Matrix(q)

    def generalized_forces(q, origin : Matrix, force : Matrix):
        '''
        Convert a force acting at origin on the model into generalized forces Q.
        '''

        return diff(origin, q).transpose()*force


    def dynamics(self, joints : list[Joint] | None = None):
        '''
        Dynamics f(q_i,dq_i,ddq_i) such that f(q_i,dq_i,ddq_i) = Q_i for each joint where Q_i are generalized forces.
        '''
        return Matrix(list(self.generate_dynamics(joints)))

    def canonical_dynamics(self, joints : list[Joint] | None = None):
        '''
        Returns tuple M(q, dq), g(q, dq) such that
        
        M(q,dq)ddq + g(dq) = external_forces.
        '''

        if joints is None:
            joints = self.joints()

        t = Symbol('t')

        q = self.q(joints)
        dq = diff(q, t)
        ddq = diff(dq, t)

        # rows of the mass matrix
        mass_rows = []

        # rest of the dynamics
        rest_rows = []
        
        for tau in self.generate_dynamics(joints):

            tau = approximate_integers(tau)
            
            # we know that tau is linear in ddq
            mass = diff(tau, ddq)

            mass_rows.append(mass)

            # again, using linearity of tau in ddq 
            rest = approximate_integers(tau.subs({ v: 0 for v in ddq}))

            rest_rows.append(rest)

        return Matrix(mass_rows), Matrix(rest_rows)
    
def inertia_matrix(ixx = 0, ixy = 0, ixz = 0, iyy = 0, iyz = 0, izz = 0):

    return Matrix([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz]
    ])
