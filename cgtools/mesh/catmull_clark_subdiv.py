from collections import defaultdict

import numpy as np
from scipy import sparse

from ..indexing import inverse_index_dict


class CatmullClarkSubdiv(object):
    def __init__(self, quads):
        quads = np.array(quads)
        assert quads.shape[1] == 4
        self._quads_lo = quads
        n_verts = quads.max() + 1

        # build edges
        def mk_edge_key(i, j):
            return (min(i, j), max(i, j))

        quads_by_edge_ij = defaultdict(list)
        edges_ij = []
        edges_by_quads = defaultdict(lambda: [None] * 4)
        quads_by_vert = defaultdict(list)
        edge_ijs_by_vert = defaultdict(list)  # {vertIndex: [(i1, j1), (i2, j2), ...]}
        for quad_ix, quad in enumerate(quads):
            for k in range(4):
                # construct edge from vertex i to j
                ij = mk_edge_key(quad[k], quad[(k + 1) % 4])
                edges_ij.append(ij)
                quads_by_edge_ij[ij].append(quad_ix)
                # build adjacency
                quads_by_vert[quad[k]].append(quad_ix)
                edge_ijs_by_vert[quad[k]].append(ij)

        # ensure all edges have 2 neighboring faces
        assert all(len(v) == 2 for v in quads_by_edge_ij.values())
        # ensure all quads reference 4 edges
        assert all(all(i is not None for i in quad) for quad in edges_by_quads.values())

        # make unique edge indices
        edges_ij_uniq = np.unique(edges_ij, axis=0)
        edge_ix_by_ij = inverse_index_dict(map(tuple, edges_ij_uniq))

        quads_hi = []
        for quad_ix, (i0, i1, i2, i3) in enumerate(quads):
            # face point index
            iF = n_verts + quad_ix
            # edge indices
            ix_offset_edges = n_verts + len(quads)
            e0 = edge_ix_by_ij[mk_edge_key(i0, i1)] + ix_offset_edges
            e1 = edge_ix_by_ij[mk_edge_key(i1, i2)] + ix_offset_edges
            e2 = edge_ix_by_ij[mk_edge_key(i2, i3)] + ix_offset_edges
            e3 = edge_ix_by_ij[mk_edge_key(i3, i0)] + ix_offset_edges
            quads_hi.append((i0, e0, iF, e3))
            quads_hi.append((e0, i1, e1, iF))
            quads_hi.append((iF, e1, i2, e2))
            quads_hi.append((e3, iF, e2, i3))

        self.quads_hi = np.array(quads_hi)

        # We're going to construct sparse matrices that perform the linear interpolation
        # of the new vertices from the old ones, see __call__ on how we can use this.
        # This will induce a certain cost to construct these matrices, but once constructed
        # the interpolation can be done very quickly.

        # construct edge interpolation matrix
        E_triplets = []
        for (i, j), edge_ix in edge_ix_by_ij.iteritems():
            q1, q2 = quads_by_edge_ij[(i, j)]
            E_triplets.append((edge_ix, i, 0.25))
            E_triplets.append((edge_ix, j, 0.25))
            E_triplets.append((edge_ix, q1 + n_verts, 0.25))
            E_triplets.append((edge_ix, q2 + n_verts, 0.25))
        E_i, E_j, E_data = zip(*E_triplets)
        self._E_intp = sparse.csr_matrix((E_data, (E_i, E_j)))

        # construct vertex interpolation matrix
        V_triplets = []
        for vert_ix in xrange(n_verts):
            n = len(quads_by_vert[vert_ix])
            # put coefficients that compute F
            adj_quads = quads_by_vert[vert_ix]
            for quad_ix in adj_quads:
                V_triplets.append((vert_ix, n_verts + quad_ix, 1 / float(len(adj_quads) * n)))
            # put coefficients that compute 2 * R
            adj_edges = edge_ijs_by_vert[vert_ix]
            for i, j in adj_edges:
                # duplicate vertices will be summed when converting to csr later
                V_triplets.append((vert_ix, i, 1 / float(len(adj_edges) * n)))
                V_triplets.append((vert_ix, j, 1 / float(len(adj_edges) * n)))
            # put coefficients to compute (n - 3) * orig
            V_triplets.append((vert_ix, vert_ix, (n - 3.) / float(n)))
        
        V_i, V_j, V_data = zip(*V_triplets)
        self._V_intp = sparse.csr_matrix((V_data, (V_i, V_j)))

    def __call__(self, attrs):
        # interpolate new face points
        face_attrs = attrs[self._quads_lo].mean(axis=1)
        # interpolate new edge points
        edge_pts = self._E_intp * np.vstack((attrs, face_attrs))
        # interpolate old vertices
        attrs_subdiv = self._V_intp * np.vstack((attrs, face_attrs))

        return np.vstack((
            attrs_subdiv, face_attrs, edge_pts
        ))
