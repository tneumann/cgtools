import numpy as np
import numpy.testing as npt
from cgtools.mesh.intersections import ray_mesh_intersect


def test_ray_mesh_intersect():
    verts = np.array([[5,5,5],[10,15,4],[15,5,3]], np.float)
    hit_tri, hit_uv, ray_ix, hit_pt = ray_mesh_intersect(
        verts,
        np.array([[0, 1, 2]], np.int32),
        np.array([[9,5,-5]], np.float),
        np.array([[0.1,0.1,0.8]], np.float),
    )
    p_true = np.array([10.121951219512194, 6.121951219512195, 3.97560975609756])
    npt.assert_equal(hit_tri, [0])
    npt.assert_equal(ray_ix, [0])
    npt.assert_allclose(hit_pt, p_true[np.newaxis, :])
    u = hit_uv[0, 0]
    v = hit_uv[0, 1]
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    npt.assert_allclose(verts[0] + u*e1 + v*e2, p_true)

