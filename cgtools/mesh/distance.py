import numpy as np
from ..vector import veclen
from .intersections import closest_points_on_mesh


__all__ = ['signed_distance']


def signed_distance(verts0, normals0, verts1):
    """ 
    Compute the euclidean distance from 3d points in verts0 to the corresponding points in verts1,
    and flip the sign of the distance corresponding to the normals in normals0.
    That is, returns negative values for dimples, positive values for bulges.
    """
    diff = verts0 - verts1
    return veclen(diff) * np.sign((normals0 * diff).sum(axis=-1))


def signed_closest_point_distance(verts0, normals0, verts1, tris1):
    """
    Compute the signed closest point distance from verts0 to verts1
    """
    _, _, hit_pts, _ = closest_points_on_mesh(verts0, verts1, tris1)
    return signed_distance(verts0, normals0, hit_pts)
