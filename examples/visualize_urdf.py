#!/bin/env python

import numpy as np
import pyvista as pv
import symengine as se

from odysseus import JointRevolute

from anymal_model import robot

model = robot.model()

q = model.q()


#
#
#
#

points = []

subs = { qi: 0 for qi in q }

joints = model.joints()

for joint in joints:

    pos = joint.get_transform().trans_.subs(subs)

    points.append([ coord.evalf() for coord in pos ])


poly = pv.PolyData(np.array(points, dtype=np.float32))

poly["Joint Names"] = [joint.name_ for joint in joints]

plotter = pv.Plotter()

plotter.add_point_labels(poly, "Joint Names", point_size=20, font_size=36)

plotter.add_mesh(poly, render_points_as_spheres=True)

plotter.show_grid()
plotter.show()
