

from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .. import vector as V
from ..indexing import filter_reindex, take_reindex
from ._igl_ext import is_border_vertex


def get_mesh_edges(verts, tris):
    u = verts[tris[:,1]] - verts[tris[:,0]]
    v = verts[tris[:,2]] - verts[tris[:,0]]
    return u, v

edges = get_mesh_edges


def get_vertex_rings(tris):
    # TODO: can we achieve this quicker with scipy.sparse?
    ring_by_vertex_id = defaultdict(list)
    edges = get_edges_from_triangles(tris)
    for i1, i2 in edges:
        ring_by_vertex_id[i1].append(i2)
        ring_by_vertex_id[i2].append(i1)
    return ring_by_vertex_id


def triangle_triangle_adjacency_list(tris):
    """
    determines triangle-triangle adjacency just from a list of triangles which are given as i,j,k index tuples
    returns an array of shape (len(edges), 2), each tuple is a pair of indices of neighboring triangles
    (in no particular order)
    """
    tri_by_edge = defaultdict(lambda: [])
    for tri_index, (i, j, k) in enumerate(tris):
        tri_by_edge[(min(i, j), max(i, j))].append(tri_index)
        tri_by_edge[(min(j, k), max(j, k))].append(tri_index)
        tri_by_edge[(min(k, i), max(k, i))].append(tri_index)
    return np.array([ts for ts in list(tri_by_edge.values()) if len(ts) == 2])


def get_edges_from_triangles(tris, directed=False):
    """
    Returns the unique edges as an array of shape (n_edges, 2)
    containing, for each edge, 2 indices of the vertices of each edge.

    If directed = True (default), then all non-boundary edges will be returned twice, 
    e.g. if there is an edge between vertex i and j, the returned array will include
    (i, j) and (j, i)

    >>> tris = [[0, 1, 2], [2, 1, 3], [3, 1, 4]]
    >>> sorted(get_edges_from_triangles(tris).tolist())
    [[0, 1], [0, 2], [1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]
    >>> tris = [[0, 1, 2], [2, 1, 3], [3, 1, 4]]
    >>> sorted(get_edges_from_triangles(tris, directed=False).tolist())
    [[0, 1], [0, 2], [1, 2], [1, 3], [1, 4], [2, 3], [3, 4]]
    """
    tris = np.array(tris)
    all_edges = tris[:, [[0,1], [1,0], [1,2], [2,1], [2,0], [0,2]]].reshape((-1, 2))
    A = sparse.coo_matrix((np.ones(len(all_edges)), all_edges.T))
    if directed:
        A = sparse.triu(A, format='csr')  # format csr necessary to remove duplicate entries in COO format
    else:
        # TODO: format csr necessary in this case?
        A = A.tocsr()
    return np.column_stack(A.nonzero())


def edge_difference_matrix(tris, directed=False):
    edges_ij = get_edges_from_triangles(tris, directed=directed)
    return sparse.csr_matrix(
        (np.tile([-1, 1], len(edges_ij)),
         (np.repeat(np.arange(len(edges_ij)), 2), edges_ij.ravel())))


def edge_adjacency_matrix(tris, n_verts=None):
    """
    Returns a scipy.sparse.csr_matrix of size n_verts x n_verts
    where element (i, j) is one if vertex i is connected to vertex j.

    If n_verts is None, it is determined automatically from the triangle array.
    """
    if n_verts is None:
        n_verts = tris.max() + 1
    ij = np.r_[np.c_[tris[:, 0], tris[:, 1]],
               np.c_[tris[:, 0], tris[:, 2]],
               np.c_[tris[:, 1], tris[:, 2]]]
    A = sparse.csr_matrix(
        (np.ones(len(ij)), ij.T),
        shape=(n_verts, n_verts))
    A = A.T + A
    A.data[:] = 1
    return A


def largest_connected_component(tris):
    """
    Returns vertex indices of the largest connected component of the mesh
    as well as a reindexed triangle array, e.g. use it like so:

    ix, tris_new = largest_connected_component(tris)
    show_mesh(verts[ix], tris_new)
    """
    _, labels = connected_components(edge_adjacency_matrix(tris), directed=False)
    largest = np.bincount(labels).argmax()
    mask = labels == largest
    vertex_indices = mask.nonzero()[0]
    tris_new = filter_reindex(mask, tris)
    return vertex_indices, tris_new


def get_per_triangle_normals(verts, tris, edges_uv=None):
    if edges_uv is None:
        u, v = get_mesh_edges(verts, tris)
    else:
        u, v = edges_uv
    return V.normalized(np.cross(u, v))


def get_triangle_frames(verts, tris, normals=None):
    u, v = get_mesh_edges(verts, tris)
    # need the normal of each triangle
    normals = normals if normals is not None else get_per_triangle_normals(verts, tris, edges_uv=(u, v))
    return np.dstack((u, v, normals)) # swapaxes???


def get_vertex_areas(verts, tris):
    PQ = verts[tris[:, 0]] - verts[tris[:, 1]]
    RP = verts[tris[:, 2]] - verts[tris[:, 0]]
    lump_area = V.veclen(np.cross(PQ, RP)) / 6.
    area = sum(np.bincount(tris[:,i], lump_area, minlength=len(verts)) for i in range(3))
    return area


def triangle_normal(verts, tris):
    u, v = edges(verts, tris)
    return np.cross(u, v)


def double_triangle_area(verts, tris):
    return V.veclen(triangle_normal(verts, tris))


def triangle_area(verts, tris):
    return double_triangle_area(verts, tris) / 2.


def quads_to_tris(quads):
    if quads.shape[1] == 3:  # already triangles?
        return quads
    return quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)


def filter_triangles(vert_ix_or_mask, tris):
    vert_ix_or_mask = np.asarray(vert_ix_or_mask)
    fn = filter_reindex if vert_ix_or_mask.dtype == np.bool else take_reindex
    return fn(vert_ix_or_mask, tris)


class ReorderByFaces(object):
    def __init__(self, faces0, faces1):
        self.ji = defaultdict(set)
        max_i = 0
        for face0, face1 in zip(faces0, faces1):
            for i, j in zip(face0, face1):
                self.ji[j].add(i)
                max_i = max(max_i, i)
        self._sz = max_i + 1
    
    def __call__(self, array):
        array_reordered = np.zeros((self._sz,) + array.shape[1:])
        for j, ixs in self.ji.iteritems():
            for i in ixs:
                # TODO: duplicating values here - is that ok or should we warn when those values are not consistent?
                array_reordered[i] = array[j]
        return array_reordered


def reorder_by_faces(faces0, faces1, array):
    return ReorderByFaces(faces0, faces1)(array)

