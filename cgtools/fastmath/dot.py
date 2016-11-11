import numpy as np
import _fastmath_ext


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
    if a.ndim == 2 and b.ndim == 3:
        # a is just a single matrix
        return np.dot(b.swapaxes(-1, -2), a.T).swapaxes(-1, -2)
    if a.ndim == 3 and b.ndim == 2:
        # b is just a single matrix
        return np.dot(a, b)
    if a.shape[-1] != b.shape[-2]:
        raise ValueError, "arrays must have suitable shape for matrix multiplication"
    a_contig = a.reshape(-1, a.shape[-2], a.shape[-1])
    b_contig = b.reshape(-1, b.shape[-2], b.shape[-1])
    return _fastmath_ext.matmat(a_contig, b_contig).reshape(a.shape[:-2] + (a.shape[-2], b.shape[-1]))


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
        # just a single matrix multiplied by multiple vectors - use numpy.dot
        return np.dot(matrices, vectors.T).T
    if vectors.ndim == 1:
        # multiple matrices multiplied by a single vector - use numpy.dot
        return np.dot(matrices, vectors)
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError, "matrices and vectors must be compatible for matrix-vector multiplication"
    matrices_contig = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1])
    vectors_contig = vectors.reshape(-1, vectors.shape[-1])
    return _fastmath_ext.matvec(matrices_contig, vectors_contig).reshape(matrices.shape[:-2] + (matrices.shape[-2], ))


if __name__ == '__main__':
    import timeit
    a = np.random.random((10000, 4, 4))
    b = np.random.random((10000, 4, 4))
    def t_matmat():
        matmat(a, b)
    def t_matmat_np():
        np.array(map(np.dot, a, b))
    print "matmat"
    print "c++:", 
    speed1 = np.mean(timeit.repeat(t_matmat, repeat=5, number=100))
    print speed1
    print "numpy:", 
    speed2 = np.mean(timeit.repeat(t_matmat_np, repeat=5, number=100))
    print speed2
    print "speedup %.2f" % np.mean(speed2 / speed1)

    a = np.random.random((10000, 4, 4))
    b = np.random.random((10000, 4))
    def t_matvec():
        matvec(a, b)
    def t_matvec_np():
        np.array(map(np.dot, a, b))
    print "matvec"
    print "c++:", 
    speed1 = np.mean(timeit.repeat(t_matvec, repeat=5, number=100))
    print speed1
    print "numpy:", 
    speed2 = np.mean(timeit.repeat(t_matvec_np, repeat=5, number=100))
    print speed2
    print "speedup %.2f" % np.mean(speed2 / speed1)

