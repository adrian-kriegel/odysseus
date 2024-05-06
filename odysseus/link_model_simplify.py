
import symengine as se

from odysseus.link_model import LinkModel, Joint, Link,  Segment

from odysseus.visit import visit_as_long

from odysseus.sym_util import approximate_integers

def merge_fixed_links(root : Link, tolerance = 1e-5) -> (se.Expr, se.Matrix, list[Joint]):
    '''
    Merge all fixed parts of the link into one.
    Returns mass, inertia, joints. Where joints are the joints the merged segment 'ends'.
    '''

    joints = []

    def predicate(link : Link):

        if not isinstance(link, Joint):
            return True
        else:
            joints.append(link)
            return False

    parts = root.visit_as_long(predicate)

    mass = 0
    inertia = se.zeros(3, 3)

    inertial_origin = root.get_transform().trans_

    segments = (part for part in parts if isinstance(part, Segment))

    for part in segments:

        t = part.get_transform(root.name_)

        rot = t.rot_
        a1,a2,a3 = t.trans_

        # rotate the intertia tensor into root's coordinate system
        J = rot * part.inertia_ * rot.transpose()

        # Steiner
        a = se.Matrix([
            [  0, -a3,  a2],
            [ a3,   0, -a1],
            [-a2,  a1,   0]  
        ])

        inertia += J + part.m_ * a.transpose()*a

        mass += part.m_

    inertia = se.Matrix([approximate_integers(i.evalf(real=True), tolerance) for i in inertia]).reshape(3, 3)

    return mass, inertia, joints

