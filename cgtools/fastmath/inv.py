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
    a = np.random.random((1000, 3, 3))
    def t_inv3():
        inv3(a)
    def t_inv3_np():
        np.array(map(np.linalg.inv, a))
    print "cython:", 
    speed1 = np.mean(timeit.repeat(t_inv3, repeat=5, number=100))
    print speed1
    print "numpy:", 
    speed2 = np.mean(timeit.repeat(t_inv3_np, repeat=5, number=100))
    print speed2
    print "speedup %.2f" % np.mean(speed2 / speed1)

