from __future__ import absolute_import

from collections import defaultdict
import numpy as np

from .. import vector as V


def get_mesh_edges(verts, tris):
    u = verts[tris[:,1]] - verts[tris[:,0]]
    v = verts[tris[:,2]] - verts[tris[:,0]]
    return u, v


def get_vertex_rings(tris):
    # TODO: can we achieve this quicker with scipy.sparse?
    ring_by_vertex_id = defaultdict(list)
    edges = get_edges_from_triangles(tris)
    for i1, i2 in edges:
        ring_by_vertex_id[i1].append(i2)
        ring_by_vertex_id[i2].append(i1)
    return ring_by_vertex_id


def get_edges_from_triangles(tris):
    """
    >>> tris = [[0, 1, 2], [2, 1, 3], [3, 1, 4]]
    >>> sorted(get_edges_from_triangles(tris).tolist())
    [[0, 1], [0, 2], [1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]
    """
    tris = np.array(tris)
    all_edges = tris[:, [[0,1], [1,2], [2,0]]].reshape((-1, 2))
    all_edges = np.column_stack((
        np.minimum(all_edges[:,0], all_edges[:,1]),
        np.maximum(all_edges[:,0], all_edges[:,1]) ))
    return np.array(list(set(map(tuple, all_edges))))


def get_per_triangle_normals(tris, verts, edges_uv=None):
    if edges_uv is None:
        u, v = get_mesh_edges(tris, verts)
    else:
        u, v = edges_uv
    return V.normalized(np.cross(u, v))


def get_triangle_frames(triangles, verts, normals=None):
    u, v = get_mesh_edges(triangles, verts)
    # need the normal of each triangle
    normals = normals if normals is not None else get_per_triangle_normals(triangles, verts, edges_uv=(u, v))
    return np.dstack((u, v, normals)) # swapaxes???
