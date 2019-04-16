import numpy as np
import numpy.testing as npt
from cgtools.mesh.intersections import ray_mesh_intersect, closest_points_on_mesh


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


def test_closest_point_on_mesh():
    verts = np.array([[5,5,5],[10,15,4],[15,5,3]], np.float)
    pts = np.array([[9,5,-5]], np.float)
    sq_dist, hit_tri, hit_xyz, hit_uv = closest_points_on_mesh(
        pts, verts, np.array([[0, 1, 2]], np.int32),
    )
    npt.assert_equal(hit_tri, [0])

    # brute force compute closest hit and compare
    u, v = list(map(np.ravel, np.mgrid[:1:1000j, :1:1000j]))
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    p = verts[np.newaxis, 0] + u[:, np.newaxis] * e1[np.newaxis] + v[:, np.newaxis] * e2[np.newaxis]
    sq_dist_brute = ((p - pts[0][np.newaxis])**2).sum(axis=1)
    imin = sq_dist_brute.argmin()
    npt.assert_almost_equal(sq_dist_brute[imin], sq_dist[0], decimal=3)
    npt.assert_almost_equal(u[imin], hit_uv[0, 0], decimal=3)
    npt.assert_almost_equal(v[imin], hit_uv[0, 1], decimal=3)
    npt.assert_equal(hit_tri, [0])


