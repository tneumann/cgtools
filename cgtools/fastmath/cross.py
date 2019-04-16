import numpy as np
from . import _fastmath_ext

__all__ = ['cross3']


def cross3(a, b):
    """
    >>> a = np.random.random((10, 3))
    >>> b = np.random.random((10, 3))
    >>> c1 = cross3(a, b)
    >>> c2 = np.cross(a, b)
    >>> np.allclose(c1, c2)
    True
    """
    a = np.asarray(a)
    b = np.asarray(b)
    orig_shape = a.shape
    if a.ndim == 1 and b.ndim == 1:
        # just a single cross product 
        return np.cross(a, b)
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("both arrays must be arrays of 3D vectors")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape, but shape(a)=%d and shape(b)=%d" % (a.shape, b.shape))
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    return _fastmath_ext.cross3(a, b).reshape(orig_shape)
