import numpy as np
from scipy import sparse

from .gradient import gradient_op
from .topology import double_triangle_area


def div_op(verts, tris):
    D = sparse.diags(np.repeat(double_triangle_area(verts, tris), 3))
    grad = gradient_op(verts, tris)
    # TODO check if 0.5 factor is correct
    #      it is 0.25 in https://github.com/alecjacobson/gptoolbox/blob/master/mesh/div.m
    #      but then we don't have the identity L = div * grad
    return -0.5 * grad.T * D


def div(verts, tris, g):
    return div_op(verts, tris) * g
