import numpy as np
from scipy import weave

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
        raise ValueError, "both arrays must be arrays of 3D vectors"
    if a.shape != b.shape:
        raise ValueError, "a and b must have the same shape, but shape(a)=%d and shape(b)=%d" % (a.shape, b.shape)
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    result = np.empty(a.shape, np.promote_types(a.dtype, b.dtype))
    n = a.shape[0]
    code = """
        using namespace blitz;
        //#pragma omp parallel for
        for(int i=0; i < n; i++) {
            const double a1 = a(i, 0);
            const double a2 = a(i, 1);
            const double a3 = a(i, 2);
            const double b1 = b(i, 0);
            const double b2 = b(i, 1);
            const double b3 = b(i, 2);
            result(i, 0) = a2 * b3 - a3 * b2;
            result(i, 1) = a3 * b1 - a1 * b3;
            result(i, 2) = a1 * b2 - a2 * b1;
        }
    """
    weave.inline(code, ['a', 'b', 'n', 'result'],
                 #verbose=2, force=1,
                 #extra_compile_args=['-fopenmp'],
                 #extra_link_args=['-fopenmp'],
                 type_converters=weave.converters.blitz)
    return result.reshape(orig_shape)

if __name__ == '__main__':
    import timeit
    a = np.random.random((17000, 3))
    b = np.random.random((17000, 3))
    np_cross = np.cross
    def t_cross3():
        cross3(a, b)
    def t_cross_np():
        np_cross(a, b)
    print "blitzcross:", 
    speed1 = np.mean(timeit.repeat(t_cross3, repeat=5, number=100))
    print speed1
    print "numpy.cross:", 
    speed2 = np.mean(timeit.repeat(t_cross_np, repeat=5, number=100))
    print speed2
    print "speedup %.2f" % np.mean(speed2 / speed1)

