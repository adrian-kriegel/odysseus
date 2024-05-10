
'''
Helpers for building LinkModels from a URDF DOM.
'''

import numpy as np
import math

from symengine import Matrix, symbols
import symengine as se

import sympy as sp

from odysseus import \
    LinkModel, \
    Joint,\
    JointRevolute, \
    Link, \
    Segment, \
    Transform,\
    inertia_matrix,\
    JointLimits

from odysseus.sym_util import approximate_integers

def find_by_attr(node, tag_name : str, attr : str, value : str):
    
    return (e for e in node.getElementsByTagName(tag_name) if e.getAttribute(attr) == value)


def approximate(x : float, tolerance=1e-4):
    '''
    Approximates x as a rational number.
    '''
    return se.sympify(sp.nsimplify(x, rational=True, tolerance=tolerance))

def approximate_angle(x : float, accuracy=1e-4):
    '''
    Approximates x as a rational multiple of pi.
    '''
    return approximate(x / np.pi * se.pi)

class URDFElement:
    '''
    Represents a tag in URDF.
    This is effectively just a wrapper around a DOM element.
    '''

    def __init__(self, node):

        self.node_ = node

    def child(self, name : str):
        '''
        Returns single immediate child as URDFElement.
        Returns None if not found.
        Raises Exception if not unique.
        '''

        matches = [child for child in self.node_.getElementsByTagName(name) if child.parentNode == self.node_]

        if len(matches) == 0:
            return None

        if len(matches) != 1:
            raise Exception(f'Child {name} of {self.tag()} is not unique!')

        return URDFElement(matches[0])

    def attr(self, name : str, fallback = None):

        value = self.node_.getAttribute(name)

        return fallback if value is None or value == '' else value


    def tag(self): 
        return self.node_.tagName
    
    def origin(self):

        origin = self.child('origin')

        return Transform() if origin is None else Transform(
            URDFElement.parse_vector(origin.attr('xyz'), lambda x: x), 
            URDFElement.parse_vector(origin.attr('rpy'), approximate_angle)
        )

    def as_sym_lm(self, parent : Link):
        '''
        Convert to the matching type from the sym_lm module.
        '''

        match self.tag():
            case 'joint':
                return self.as_joint(parent)

            case 'link':
                return self.as_link(parent)

    def as_link(self, parent : Link) -> Segment | Link:
        '''
        Parse <link> as Segment or Link
        '''

        inertial = self.child('inertial')
        
        if inertial is None:
            return Link(self.attr('name'), parent, self.origin())
        else:
            inertia_element = inertial.child('inertia')
            mass_element = inertial.child('mass')

            inertia_values = {}

            if inertia_element:
                inertia_values = { k: approximate_integers(float(v)) for k,v in inertia_element.node_.attributes.items() }

            if mass_element:
                mass = approximate_integers(float(mass_element.attr('value', 0)), 1e-5)
            else:
                mass = 0

            inertia = inertia_matrix(**inertia_values)

            if inertia == inertia_matrix() and mass == 0:
                # a Segment is by definition a Link with mass/inertia
                return Link(self.attr('name'), parent, self.origin())
            
            return Segment(
                self.attr('name'), 
                parent, 
                self.origin(), 
                mass,
                inertia
            )

    def limits(self) -> JointLimits:

        limits_element = self.child('limit')

        if limits_element is None:
            return JointLimits()
        else:
            return JointLimits(
                float(limits_element.attr('lower', -np.inf)),
                float(limits_element.attr('upper', np.inf)),
                - float(limits_element.attr('velocity', np.inf)),
                float(limits_element.attr('velocity', np.inf))
            )

    def as_joint(self, parent : Link):
        '''
        Parse <joint> tag as Joint (w/ dof) or Link (w/o dof).
        '''

        limits = self.limits()

        match self.attr('type'):
            case 'revolute':
                return JointRevolute(
                    self.attr('name'),
                    parent,
                    self.origin(),
                    URDFElement.parse_vector(self.child('axis').attr('xyz')),
                    float(self.attr('damping', 0)),
                    limits=limits
                )
            case 'fixed':
                # a fixed joint is just a link
                return Link(
                    self.attr('name'),
                    parent,
                    self.origin()
                )
            case joint_type:
                raise Exception(f'Unsupported joint type: {joint_type}.')

    @staticmethod 
    def parse_vector(v : str | None, map_func=approximate) -> Matrix:
        ''' 
        Parse a 3D vector formatted as a space separated string.
        Returns zero-vector if v is None.
        '''
        
        if v is not None:
            return Matrix([map_func(float(val)) for val in v.split(' ')])
        else:
            return Matrix([0,0,0])

class URDFModel:
    '''
    Helper class for building a LinkModel from a URDF file.
    '''

    def __init__(self, dom):

        '''
        Keyword arguments:
        dom         -- Parsed URDF file (e.g. xml.dom.minidom.parse).
        origin      -- Optional origin for the entire robot assembly.
        '''

        # parsed URDF document
        self.dom_ = dom
        
        # resulting LinkModel 
        self.model_ = LinkModel()

        self.transform_element_ = lambda element, parent: element.as_sym_lm(parent)

    def transform_elements(self, callback):
        '''
        Override the behavior of turning URDFElement into a Link object.
        '''
        self.transform_element_ = callback

    def model(self) -> LinkModel:
        return self.model_

    def add_element_by_name(self, name : str, parent = None):
        '''
        Add a <link> or <joint> froom the URDF and all of its children to the LinkModel.
        '''

        self.add_element(self.find_by_name(name, '*'), parent)

    def add_element(self, element : URDFElement, parent = None):
        '''
        Add a link/joint froom the URDF and all of its children to the LinkModel.
        '''

        # link will add itself to self.model_ if passed a parent

        link = self.transform_element_(element, parent)

        if link is None:
            link = element.as_sym_lm(parent)
        
        if element.tag() == 'joint':
            child = element.child('child').attr('link')
            
            self.add_element(self.find_link(child), link)
        else:
            for joint in self.find_joints_for_parent(link.name_):
                self.add_element(joint, link)

    def find_joints_for_parent(self, name : str):
        '''
        Generator for all joints attached to the link with the specified name.
        '''

        return (
            URDFElement(joint) 
            for joint in self.dom_.getElementsByTagName('joint') if joint.parentNode.tagName == 'robot'
            if joint.getElementsByTagName('parent').item(0).getAttribute('link') == name
        )

    def find_by_name(self, name : str, type : str, ReturnType = URDFElement):
        '''
        Find any element in the URDF by name.
        type        -- Tag name or '*'
        name        -- Optional origin for the entire robot assembly.
        '''

        node = next(find_by_attr(self.dom_, type, 'name', name), None)

        if node is None:
            raise Exception(f'Could not find {name}')

        return ReturnType(node)

    def find_link(self, name : str):
        '''
        Find <link> by name.
        '''

        return self.find_by_name(name, 'link')

    def find_joint(self, name : str):
        '''
        Find <joint> by name.
        '''

        return self.find_by_name(name, 'joint')
    