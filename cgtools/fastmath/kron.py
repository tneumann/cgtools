import numpy as np
from . import _fastmath_ext


__all__ = ['multikron']


def multikron(a, b):
    """
    parallel kronecker product over arrays of stacked matrices
    (e.g. performs np.kron over the last 2 dimensions of a and b)

    >>> a = np.random.random((10, 3, 5))
    >>> b = np.random.random((10, 5, 4))
    >>> r1 = multikron(a, b)
    >>> r2 = np.array(list(map(np.kron, a, b)))
    >>> np.allclose(r1, r2)
    True
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2 and b.ndim == 2:
        # just a single matrix vector multiplication
        return np.kron(a, b)
    if a.ndim == 2 and b.ndim == 3:
        # a is just a single matrix
        raise NotImplementedError
    if a.ndim == 3 and b.ndim == 2:
        # b is just a single matrix
        raise NotImplementedError
    if a.shape[:-2] != b.shape[:-2]:
        b = np.broadcast_to(b, a.shape[:-2] + b.shape[-2:])
    a_contig = a.reshape(-1, a.shape[-2], a.shape[-1])
    b_contig = b.reshape(-1, b.shape[-2], b.shape[-1])
    if a_contig.shape[0] != b_contig.shape[0]:
        raise ValueError("array shapes are not compatible: %s vs %s. Expect shapes to be compatible up to the last 2 axes" % (a.shape, b.shape))
    return _fastmath_ext.multikron(a_contig, b_contig).reshape(a.shape[:-2] + (a.shape[-2]*b.shape[-2], a.shape[-1]*b.shape[-1]))

