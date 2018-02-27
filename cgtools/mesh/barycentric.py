import numpy as np
from scipy import sparse

from topology import get_mesh_edges


def barycentric_matrix(uv, tris, num_verts):
    """
    Return the barycentric coordinate matrix B such that

        B * verts = verts_new

    where verts_new yield the barycentric interpolation according to uv
    of the triangles given as n*3 index array in tris.

    Given the barycentric coordinates u and v of a triangle 
    with points p1, p2, p3 (given by indices i1, i2, i3),
    barycentric interpolation yields the new point pb

        pb = p1 + u * (p2 - p1) + v * (p3 - p1)

    """
    uvw = np.column_stack((1 - uv[:,0] - uv[:,1], uv[:,0], uv[:,1]))
    return sparse.csr_matrix((uvw.ravel(), 
                              (np.mgrid[:len(uvw), :3][0].ravel(),
                               tris.ravel()) ), 
                             shape=(len(uvw), num_verts))


def barycentric_interpolate(verts, tris, uv):
    """
    Compute 3d points from a given set of barycentric coordinates

    Given the barycentric coordinates u and v of a triangle in uv
    with points p1, p2, p3 (given by indices i1, i2, i3 in tris),
    barycentric interpolation yields the new point pb

        pb = p1 + u * (p2 - p1) + v * (p3 - p1)
    """
    edge1, edge2 = get_mesh_edges(tris, verts)
    return verts[tris[:,0]] + \
            uv[:,0][:,np.newaxis] * edge1 + \
            uv[:,1][:,np.newaxis] * edge2
