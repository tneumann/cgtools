import numpy as np
from scipy import weave


__all__ = ['matmat', 'matvec']

def matmat(a, b):
    """
    >>> a = np.random.random((10, 3, 5))
    >>> b = np.random.random((10, 5, 4))
    >>> r1 = matmat(a, b)
    >>> r2 = np.array(map(np.dot, a, b))
    >>> np.allclose(r1, r2)
    True
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2 and b.ndim == 2:
        # just a single matrix vector multiplication, no need to use blitzdot here
        return np.dot(a, b)
    if a.ndim == 2:
        a = a[np.newaxis]
    if b.ndim == 2:
        b = b[np.newaxis]
    if a.shape[2] != b.shape[1]:
        raise ValueError, "arrays must have suitable shape for matrix multiplication"
    if a.ndim != b.ndim != 3:
        raise ValueError, "both arrays are expected to be arrays of 2d matrices (thus, should have 3 dimensions)"
    if a.shape[-1] != b.shape[-2] or a.shape[-2] != b.shape[-1]:
        raise ValueError, "matrices must be compatible for multiplication"
    code = """
        using namespace blitz;
        firstIndex i;
        secondIndex j;
        thirdIndex k;
        fourthIndex l;
        result = sum(a(i,j,l) * b(i,l,k), l);

    """
    result = np.empty((max(a.shape[0], b.shape[0]), 
                       a.shape[1], b.shape[2]), np.promote_types(a.dtype, b.dtype))
    weave.inline(code,
                 ["a", "b", "result"],
                 type_converters=weave.converters.blitz,
                 extra_compile_args=['-w'],
                 )
    return result

def matvec(matrices, vectors):
    """
    >>> a = np.random.random((10, 3, 4))
    >>> b = np.random.random((10, 4))
    >>> r1 = matvec(a, b)
    >>> r2 = np.array(map(np.dot, a, b))
    >>> np.allclose(r1, r2)
    True
    """
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError, "vertices and matrices should have same dimension"
    if matrices.ndim == 2 and vectors.ndim == 1:
        # just a single matrix vector multiplication, no need to use blitzdot here
        return np.dot(matrices, vectors)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis]
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis]
    if matrices.ndim != 3:
        raise ValueError, "the matrices argument is expected to be an array of multiple matrices (thus, should have 3 dimensions)"
    if vectors.ndim != 2:
        raise ValueError, "the vectors argument is expected to be an array of multiple vectors (thus, should have 2 dimensions)"
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError, "matrices and vectors must be compatible for matrix-vector multiplication"
    code = """
        using namespace blitz;
        firstIndex i;
        secondIndex j;
        thirdIndex k;
        fourthIndex l;
        result = sum(matrices(i,j,k) * vectors(i,k), k);

    """
    result = np.empty((max(matrices.shape[0], vectors.shape[0]), matrices.shape[1]),
                      np.promote_types(matrices.dtype, vectors.dtype))
    weave.inline(code,
                 ["matrices", "vectors", "result"],
                 type_converters=weave.converters.blitz,
                 extra_compile_args=['-w'])
    return result


if __name__ == '__main__':
    import timeit
    a = np.random.random((10000, 4, 4))
    b = np.random.random((10000, 4, 4))
    def t_matmat():
        matmat(a, b)
    def t_matmat_np():
        np.array(map(np.dot, a, b))
    print "blitzdot:", 
    speed1 = np.mean(timeit.repeat(t_matmat, repeat=5, number=100))
    print speed1
    print "numpy:", 
    speed2 = np.mean(timeit.repeat(t_matmat_np, repeat=5, number=100))
    print speed2
    print "speedup %.2f" % np.mean(speed2 / speed1)

