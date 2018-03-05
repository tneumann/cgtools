import numpy as np
import _fastmath_ext

__all__ = ['inv2', 'inv3']


def inv3(matrices):
    if matrices.shape[-2:] != (3, 3):
        raise ValueError, "Can only invert 3x3 matrices"
    Ts = matrices.reshape((-1, 3, 3))
    Qs = _fastmath_ext.inv3(Ts)
    return Qs.reshape(matrices.shape).astype(matrices.dtype)

def inv2(matrices):
    if matrices.shape[-2:] != (2, 2):
        raise ValueError, "Can only invert 2x2 matrices"
    Ts = matrices.reshape((-1, 2, 2))
    Qs = _fastmath_ext.inv2(Ts)
    return Qs.reshape(matrices.shape)


if __name__ == '__main__':
    import timeit
    for dim in [2, 3]:
        print("testing inv%d" % dim)
        a = np.random.random((1000, dim, dim))
        inv_func = inv2 if dim == 2 else inv3
        def t_cgtools():
            inv_func(a)
        def t_np():
            np.array(map(np.linalg.inv, a))
        speed1 = np.mean(timeit.repeat(t_cgtools, repeat=5, number=100))
        print("  cgtools: %f" % speed1)
        speed2 = np.mean(timeit.repeat(t_np, repeat=5, number=100))
        print("  numpy: %f" % speed2)
        print("  speedup %.2f" % np.mean(speed2 / speed1))

