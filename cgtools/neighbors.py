from __future__ import print_function

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import networkx as nx


__all__ = ['closest_points', 'optimal_permutation']


def closest_points(from_pts, to_pts):
    return cKDTree(to_pts).query(from_pts)[1]


def optimal_permutation(from_pts, to_pts, distance_metric='euclidean'):
    """ 
    Compute optimal matching between point sets, 
    so that any point in from_pts will be matched to at most one point in to_pts
    and vice-versa.

    Returns the matching as an array of shape (n, 2) with pairs of indices
    for all correspondences, 
    e.g. [(0, 2), (3, 4)] means point 0 in from_pts matches point 2 in to_pts
    and point 3 in from_pts matches point 4 in to_pts
    """
    # build distance matrix
    D = cdist(from_pts, to_pts, distance_metric)
    Dmax = D.max()
    # construct bipartite graph and compute max weighted matching
    g = nx.Graph()
    for i in range(from_pts.shape[0]):
        for j in range(to_pts.shape[0]):
            w = Dmax - D[i, j]
            g.add_edge(('from', i), ('to', j), weight=w)
    mates = nx.max_weight_matching(g, True)
    # read out permutation
    ij = []
    for (which1, ix1), (which2, ix2) in mates:
        if which1 == 'from':
            ij.append((ix1, ix2))
        else:
            ij.append((ix2, ix1))
    return np.array(ij)