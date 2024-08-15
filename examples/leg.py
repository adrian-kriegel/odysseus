
from symengine import Matrix, symbols, Symbol, Function
import symengine as se
from symengine.printing import CCodePrinter

import casadi as ca

from odysseus import LinkModel, Segment, JointRevolute, Transform, inertia_matrix, diff

from sym_util import subs_with_indexed, flatten_matrix, approximate_integers

from generate_nmpc import generate_pl_nmpc

model = LinkModel()

x = Function('x')('t')
y = Function('y')('t')
z = Function('z')('t')
rx = Function('rx')('t')
ry = Function('ry')('t')
rz = Function('rz')('t')

root = model.create_root(
    'root', 
    Matrix((x,y,z)), 
    Matrix((rx, ry, rz))
)

j0 = JointRevolute('j0', root, axis=Matrix([1,0,0]), origin=Transform())

s0 = Segment('s0', j0, Transform(Matrix([1,0,0])), 1, inertia_matrix(0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

j1 = JointRevolute('j1', s0, axis=Matrix([0,1,0]), origin=Transform())

s1 = Segment('s1', j1, Transform(Matrix([1,0,0])), 1, inertia_matrix(0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

j2 = JointRevolute('j2', s1, axis=Matrix([0,1,0]), origin=Transform())

s2 = Segment('s2', j2, Transform(Matrix([1,0,0])), 1, inertia_matrix(0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

j3 = JointRevolute('j3', s2, axis=Matrix([0,1,0]), origin=Transform())

s3 = Segment('s3', j3, Transform(Matrix([1,0,0])), 1, inertia_matrix(0.1, 0.1, 0.1, 0.1, 0.1, 0.1))


joints_dynamics = [root, j0]

joints_external = [joint for joint in model.joints() if not joint in joints_dynamics]

# get all coordinates of the model
q = model.q(joints_dynamics)
# these are externally controlled coordinates
q_ext = model.q(joints_external)
dq_ext = diff(q_ext, 't')
ddq_ext = diff(dq_ext, 't')

# mass matrix and residual term
mass, rest = model.canonical_dynamics(joints_dynamics)

def generate_ccode():
    mass = subs_with_indexed(mass, q, 'q')
    mass = subs_with_indexed(mass, q_ext, 'q_ext')

    rest = subs_with_indexed(rest, q, 'q')
    rest = subs_with_indexed(rest, q_ext, 'q_ext')

    # flatten mass matrix for symengine cse (symengine cse cannot deal with Matrix type)
    # flatten mass matrix for symengine cse (symengine cse cannot deal with Matrix type)
    mass = list(flatten_matrix(mass))
    rest = list(rest)

    # remove any terms that are essentially integers (especially zeros)
    exprs = approximate_integers([approximate_integers(expr) for expr in mass + rest])


    # replace common subexpressions
    subexprs, exprs = se.cse(exprs)

    ccp = CCodePrinter()

    subexpr_assign = '\n    '.join([f'const double {name} = {ccp.doprint(expr)};' for name, expr in subexprs])

    # split up the simplified expressions again as we have merged them for cse
    mass_assign = '\n    '.join([f'mass_out[{i}] = {ccp.doprint(exprs[i])};' for i in range(len(mass)) ])
    rest_assign = '\n    '.join([f'rest_out[{i}] = {ccp.doprint(exprs[i])};' for i in range(len(mass), len(exprs)) ])

    input_assign = []

    # symengine does not allow us to name symbols q[0] ... :-(
    for i in range(len(q)):
        input_assign.append(f'const double& q_{i} = q[{i}];')
        input_assign.append(f'const double& dq_{i} = dq[{i}];')
        input_assign.append(f'const double& ddq_{i} = ddq[{i}];')


    for i in range(len(q)):
        input_assign.append(f'const double& q_ext_{i} = q_ext[{i}];')
        input_assign.append(f'const double& dq_ext_{i} = dq_ext[{i}];')
        input_assign.append(f'const double& ddq_ext_{i} = ddq_ext[{i}];')

    input_assign = '\n    '.join(input_assign)

    code = f'''

    #include <cmath>

    void calc_dynamics(
        double* mass_out,
        double* rest_out,
        double* ddq,
        double* dq,
        double* q,
        double* ddq_ext,
        double* dq_ext,
        double* q_ext
    ) {{

        const double g = 9.81;

        {input_assign}

        {subexpr_assign}

        {mass_assign}

        {rest_assign}
    }}
    '''

    open('generated.c', 'w').write(code)

print('casadi...')

# ignoring external influences
subs_q_ext = { q_ext_i: 0 for q_ext_i in list(q_ext)+list(dq_ext)+list(ddq_ext) } | { Symbol('g'): 9.81 }

mass = mass.subs(subs_q_ext)
rest = rest.subs(subs_q_ext)

foot_z = j3.get_transform().trans_[2]

target_pose = Matrix(symbols('t_x t_y t_z t_rx t_ry t_rz'))

robot_pose = Matrix([x, y, z, rx, ry, rz])

weight = se.eye(6)

pose_error = robot_pose - target_pose

objective = (pose_error.transpose() * weight * pose_error)

generate_pl_nmpc(
    q,
    mass
    rest,
    n_passive=6,
    state_constraint=Matrix([foot_z]),
    state_objective=objective
    ocp_parameters=[target_pose]
    extra_substitutions=subs_q_ext,
)

# generate_ccode()