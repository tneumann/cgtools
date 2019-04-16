import numpy as np
from . import _fastmath_ext

__all__ = ['inv2', 'inv3']


def inv3(matrices):
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("Can only invert 3x3 matrices")
    Ts = matrices.reshape((-1, 3, 3))
    Qs = _fastmath_ext.inv3(Ts)
    return Qs.reshape(matrices.shape).astype(matrices.dtype)

def inv2(matrices):
    if matrices.shape[-2:] != (2, 2):
        raise ValueError("Can only invert 2x2 matrices")
    Ts = matrices.reshape((-1, 2, 2))
    Qs = _fastmath_ext.inv2(Ts)
    return Qs.reshape(matrices.shape)
