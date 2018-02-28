import numpy as np
import numpy.testing as npt
from cgtools.vis.mesh import compute_normals

def test_compute_normals():
    pts = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 0)], np.float)
    tris = np.array([(0, 1, 2)], np.int)
    normals = compute_normals(pts, tris)
    npt.assert_allclose(normals, [(0, 0, 1), (0, 0, 1), (0, 0, 1)])

