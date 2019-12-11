from scipy.sparse.linalg import factorized

from .topology import get_triangle_frames
from .div import div_op
from .laplacian import compute_mesh_laplacian
from ..fastmath import matmat, inv3


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
    s = DefGradSolver(verts_tgt, tris)
    return s.transfer(verts_src, verts_src_deformed)


class DefGradSolver(object):
    def __init__(self, verts_template, tris):
        self.v0 = verts_template
        self.tris = tris
        # setup poisson system - see Chapter 5 in Botsch et al. 2006
        L = compute_mesh_laplacian(self.v0, self.tris, weight_type='cotangent',
                                   area_type='lumped_mass', return_vertex_area=False)
        self.solve = factorized(L.tocsc())
        self.div = div_op(self.v0, self.tris)

    def reconstruct(self, defgrads, align_to_template=True):
        """ reconstruct from array of (num_tris, 3, 3), return vertex coordinates """
        rhs = defgrads.transpose(0, 2, 1).reshape(-1, 3)
        # solve poisson system - see Chapter 5 in Botsch et al. 2006
        verts = self.solve(self.div * rhs)
        if align_to_template:
            verts -= (verts - self.v0).mean(axis=0)
        return verts

    def transfer(self, verts_src, verts_src_deformed, **kwargs):
        """ 
        computes the deformation between verts_src and verts_src_deformed
        and applies this deformation to the verts_template passed to the constructor
        of this DefgradSolver
        """
        # D contains deformation gradients for each triangle: (n_triangles, 3, 3)
        D = defgrads(verts_src, verts_src_deformed, self.tris)
        return self.reconstruct(D, **kwargs)
