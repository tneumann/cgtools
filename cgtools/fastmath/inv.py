import numpy as np
from scipy import weave
import _fastmath

__all__ = ['inv2', 'inv3']


def inv3(matrices):
    if matrices.shape[-2:] != (3, 3):
        raise ValueError, "Can only invert 3x3 matrices"
    Ts = matrices.reshape((-1, 3, 3))
    # TODO: cython function is not specialized for float32 - maybe use fused types?
    Qs = _fastmath.inv3(Ts.astype(np.float64))
    return Qs.reshape(matrices.shape).astype(matrices.dtype)

def inv2(matrices):
    if matrices.shape[-2:] != (2, 2):
        raise ValueError, "Can only invert 2x2 matrices"
    Ts = matrices.reshape((-1, 2, 2))
    Qs = np.empty_like(Ts) # result
    num_matrices = len(Ts)
    if Ts.dtype == np.float32:
        float_typename = "float"
    elif Ts.dtype == np.float:
        float_typename = "double"
    else:
        raise ValueError, "can only accept matrices of type float32 or float64 (double)"
    code = """
    using namespace blitz;

    for(unsigned int i=0; i<num_matrices; i++) {
        Array<%(float)s, 2> T = Ts(i, Range::all(), Range::all());
        double det = T(0,0) * T(1, 1) - T(0, 1) * T(1, 0);
        double invDet = 1. / det;
        Array<%(float)s, 2> Q = Qs(i, Range::all(), Range::all());
        Q(0, 0) = T(1, 1) * invDet;
        Q(0, 1) = -1 * T(0, 1) * invDet;
        Q(1, 0) = -1 * T(1, 0) * invDet;
        Q(1, 1) = T(0, 0) * invDet;
    }
    """ % {'float': float_typename}
    weave.inline(code,
                 ["Ts", "Qs", "num_matrices"],
                 type_converters=weave.converters.blitz,
                 extra_compile_args=['-w'])
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

