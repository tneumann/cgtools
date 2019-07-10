import numpy as np
from . import _fastmath_ext


__all__ = ['polar_dec']


def polar_dec(matrices):
    """
    Batched polar decomposition of an array of stacked matrices,
    e.g. given matrices [M1, M2, ..., Mn], decomposes each matrix
    into rotation and skew-symmetric matrices.

    >>> matrices = np.random.random((10, 3, 3))
    >>> rotations, stretches = polar_dec(matrices)
    >>> np.allclose([np.linalg.det(R) for R in rotations], 1.0)
    True
    """
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis]
        single_matrix = True
    else:
        single_matrix = False
    Rs, Ss = _fastmath_ext.polar_dec(matrices)
    if single_matrix:
        return Rs[0], Ss[0]
    else:
        return Rs, Ss
