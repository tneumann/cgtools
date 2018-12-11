import numpy as np
from scipy.sparse.linalg import spsolve

from .topology import get_triangle_frames
from .div import div
from .laplacian import compute_mesh_laplacian
from ..vector import normalized
from ..fastmath.dot import matmat
from ..fastmath.inv import inv3


def defgrads(verts_src, verts_deformed, tris):
    """ 
    Compute deformation gradients between source and deformed mesh
    """
    S0 = get_triangle_frames(verts_src, tris)
    S1 = get_triangle_frames(verts_deformed, tris)
    return matmat(S1, inv3(S0))


def deformation_transfer(verts_src, verts_src_deformed, verts_tgt, tris):
    """ 
    Deforms vertices given in "verts_tgt" so that their deformation
    matches the deformation seen between "verts_src" and "verts_deformed".

    This method assumes common topology (same triangles) in all 3 meshes.

    Returns the new vertex coordinates.

    Implemented as described in
        "Deformation Transfer for Detail-Preserving Surface Editing"
        Mario Botsch, Robert W. Sumner, Mark Pauly, Markus Gross
        in VMV 2006

    """
    # D contains deformation gradients for each triangle: (n_triangles, 3, 3)
    D = defgrads(verts_src, verts_src_deformed, tris)
    # setup & solve poisson system - see Chapter 5 in Botsch et al. 2006
    L = compute_mesh_laplacian(verts_tgt, tris,  weight_type='cotangent',
                               area_type='lumped_mass', return_vertex_area=False)
    rhs = D.transpose(0, 2, 1).reshape(-1, 3)
    verts = spsolve(L, div(verts_tgt, tris, rhs))
    # resolve translational ambiguity
    verts -= (verts - verts_tgt).mean(axis=0)
    return verts
