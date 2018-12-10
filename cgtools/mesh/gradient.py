import numpy as np
from scipy import sparse

from .. import vector as V


def gradient_op(verts, tris):
    """Build sparse matrix operator that computes 
    the gradient of a scalar function f defined on 
    a triangular mesh, e.g. returns G such that:

        (G * f).reshape(-1, 3) 

    is the gradient of f on the mesh.

    The gradient is computed according to:
        grad f = 1/2A * normal x (f_i * e_jk + f_j * e_ki + f_k * e_ij)
    For derivation see http://dgd.service.tu-berlin.de/wordpress/vismathws10/2012/10/17/gradient-of-scalar-functions/
    or in Crane et al. 2013 "Geodesics in Heat" (Sec. 3.2).
    
    Arguments:
        verts -- vertices, shape (n, 3)
        tris -- index array of triangles, shape (m, 3)
    
    Returns:
        sparse.csr_matrix -- sparse matrix G with shape (3*m, n)
    """
    n_verts = verts.shape[0]
    n_tris = tris.shape[0]
    # alias for indices of vertices
    i = tris[:, 0]
    j = tris[:, 1]
    k = tris[:, 2]
    e_ij = verts[j] - verts[i]
    e_jk = verts[k] - verts[j]
    e_ki = verts[i] - verts[k]
    normal = np.cross(e_ij, e_jk)
    double_area_sq = V.sq_veclen(normal)
    # row index:
    # [0, 0, 0, 1, 1, 1, ....]
    row = np.repeat(np.arange(3*n_tris), 3)
    # column index
    # [i[0], j[0], k[0], i[0], j[0], k[0], i[0], j[0], k[0], 
    #  i[1], j[1], k[1], i[1], j[1], k[1], i[1], j[1], k[1], 
    #  ... ]
    col = np.tile(tris, 3).ravel()
    # values are the cross products of the normal with the opposite edge
    val = (np.dstack((
        np.cross(normal, e_jk), # vertex i
        np.cross(normal, e_ki), # vertex j
        np.cross(normal, e_ij), # vertex k
    )) / double_area_sq[:, np.newaxis, np.newaxis]).ravel()

    G = sparse.csr_matrix((val, (row, col)), 
                          shape=(3*n_tris, n_verts))
    return G


def gradient(verts, tris, f):
    return (gradient_op(verts, tris) * f).reshape(-1, 3)


# There are alternative formulations:
# https://github.com/alecjacobson/gptoolbox/blob/master/mesh/grad.m
# also in http://www.hao-li.com/cs599-ss2015/slides/Lecture04.1.pdf
# I tested those and up to numerical imprecision, their result
# was the same. Here is the commented-out code for anyone interested:
#
# def gradient_op(verts, tris):
#     n_tris = tris.shape[0]
#     n_verts = verts.shape[0]
#     i1 = tris[:, 0]
#     i2 = tris[:, 1]
#     i3 = tris[:, 2]
#     v32 = verts[i3] - verts[i2]  
#     v13 = verts[i1] - verts[i3] 
#     v21 = verts[i2] - verts[i1]
#     n  = np.cross(v32, v13)
#     dblA = V.veclen(n)
#     u = V.normalized(n)
#     eperp21 = np.cross(u, v21) / dblA[:, np.newaxis]
#     eperp13 = np.cross(u, v13) / dblA[:, np.newaxis]

#     row = np.repeat(np.arange(3*n_tris), 4)
#     col = np.tile(tris[:, [1, 0, 2, 0]], 3).ravel()
#     val = np.dstack((
#         eperp13, -eperp13, eperp21, -eperp21
#     )).ravel()
#     G = sparse.csr_matrix((val, (row, col)), 
#                           shape=(3*n_tris, n_verts))
#     return G


if __name__ == "__main__":
    from os import path
    from mayavi import mlab
    from scipy.sparse.linalg import eigsh

    from ..io import load_mesh
    from ..vis.mesh import vismesh
    from .laplacian import compute_mesh_laplacian
    
    verts, tris = load_mesh(path.join(path.dirname(__file__), 'bunny_2503.obj'))

    # we need an example function defined on the surface
    # lets take an eigenvector of the laplacian of the mesh
    L, va = compute_mesh_laplacian(verts, tris)
    _, eigvecs = eigsh(-L, M=sparse.diags(va), k=64, sigma=0)
    f = eigvecs[:, -1]

    # compute gradient of heat flow and visualize
    if 0:
        grad_f = gradient(verts, tris, f)
    else:
        G = gradient_op2(verts, tris)
        grad_f = (G * f).reshape(-1, 3)


    vismesh(verts, tris, scalars=f)

    midpts = verts[tris].mean(axis=1)
    mlab.quiver3d(
        midpts[:, 0], midpts[:, 1], midpts[:, 2], 
        grad_f[:, 0], grad_f[:, 1], grad_f[:, 2],
        mode='2darrow', color=(0, 0, 0), line_width=1,
    )

    mlab.show()
