import numpy as np
from scipy import sparse

from ..vector import veclen, normalized, sq_veclen
from .laplacian import compute_mesh_laplacian
from .gradient import gradient_op
from .div import div_op

try:
    from sksparse.cholmod import cholesky
    factorized = lambda A: cholesky(A, mode='simplicial')
except ImportError:
    print "CHOLMOD not found - trying to use slower LU factorization from scipy"
    print "install scikits.sparse to use the faster cholesky factorization"
    from scipy.sparse.linalg import factorized


class GeodesicDistanceComputation(object):
    """
    Computation of geodesic distances on triangle meshes using the heat method from the impressive paper

        Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow
        Keenan Crane, Clarisse Weischedel, Max Wardetzky
        ACM Transactions on Graphics (SIGGRAPH 2013)

    Example usage:
        $ compute_distance = GeodesicDistanceComputation(vertices, triangles)
        $ distance_of_each_vertex_to_vertex_0 = compute_distance(0)

    """

    def __init__(self, verts, tris, m=1.0):
        self._verts = verts
        self._tris = tris
        # precompute some stuff needed later on
        self._grad = gradient_op(verts, tris)
        self._div = div_op(verts, tris)
        e01 = verts[tris[:,1]] - verts[tris[:,0]]
        e12 = verts[tris[:,2]] - verts[tris[:,1]]
        e20 = verts[tris[:,0]] - verts[tris[:,2]]
        # parameters for heat method
        h = np.mean(map(veclen, [e01, e12, e20]))
        t = m * h ** 2
        # pre-factorize poisson systems
        Lc, vertex_area = compute_mesh_laplacian(verts, tris, area_type='lumped_mass')
        # TODO: could actually compute: Lc = self._div * self._grad
        A = sparse.spdiags(vertex_area, 0, len(verts), len(verts))
        #self._factored_AtLc = splu((A - t * Lc).tocsc()).solve
        self._factored_AtLc = factorized((A - t * Lc).tocsc())
        #self._factored_L = splu(Lc.tocsc()).solve
        self._factored_L = factorized(Lc.tocsc())

    def __call__(self, idx):
        """
        computes geodesic distances to all vertices in the mesh
        idx can be either an integer (single vertex index) or a list of vertex indices
        or an array of bools of length n (with n the number of vertices in the mesh) 
        """
        u0 = np.zeros(len(self._verts))
        u0[idx] = 1.0
        # -- heat method, step 1
        u = self._factored_AtLc(u0).ravel()
        # running heat flow with multiple sources results in heat flowing
        # into the source region. So just set the source region to the constrained value.
        u[idx] = 1
        # I tried solving the equality-constrained quadratic program that would fix this
        # during the solve, but that did not seem to yield a lower error 
        # (but it meant that prefactorization is not straightforward)
        # The QP solution would look something like:
        #    from scipy import sparse
        #    from cgtools.indexing import sparse_indicator_matrix
        #    I = sparse_indicator_matrix(idx, self._verts.shape[0])
        #    Q = sparse.bmat([(self.A - self.t * self.Lc, I.T),
        #                    (I, None)])
        #    u = sparse.linalg.spsolve(Q, np.concatenate((u0, np.ones(I.shape[0]))))[:self._verts.shape[0]]

        # -- heat method step 2: compute gradients & normalize
        # additional normalization accross triangles helps overall numerical stability
        n_u = 1. / (u[self._tris].sum(axis=1))
        # compute gradient
        grad_u = (self._grad * u).reshape(-1, 3) * n_u[:, np.newaxis]
        # normalize gradient
        with np.errstate(all='ignore'):
            X = grad_u / veclen(grad_u)[:, np.newaxis]
            X = np.nan_to_num(X, copy=False)

        # -- heat method step 3: solve poisson system
        div_Xs = self._div * X.ravel()
        phi = self._factored_L(div_Xs).ravel()
        # transform to distances
        phi = phi - phi.min()
        phi = phi.max() - phi
        return phi

