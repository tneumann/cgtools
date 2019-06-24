import numpy as np
from . import _fastmath_ext


__all__ = ['matmat', 'matvec']


def matmat(a, b):
    """
    >>> a = np.random.random((10, 3, 5))
    >>> b = np.random.random((10, 5, 4))
    >>> r1 = matmat(a, b)
    >>> r2 = np.array(list(map(np.dot, a, b)))
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
        raise ValueError("arrays (shapes %s and %s)must have suitable shape for matrix multiplication" % (a.shape, b.shape))

    if a.shape[:-2] != b.shape[:-2]:
        shp = np.broadcast(a[..., 0, 0], b[..., 0, 0]).shape
        a_bc = np.broadcast_to(a, shp + a.shape[-2:])
        b_bc = np.broadcast_to(b, shp + b.shape[-2:])
    else:
        a_bc, b_bc = a, b
    a_contig = a_bc.reshape(-1, a.shape[-2], a.shape[-1])
    b_contig = b_bc.reshape(-1, b.shape[-2], b.shape[-1])
    return _fastmath_ext.matmat(a_contig, b_contig).reshape(a_bc.shape[:-2] + (a.shape[-2], b.shape[-1]))


def matvec(matrices, vectors):
    """
    >>> a = np.random.random((10, 3, 4))
    >>> b = np.random.random((10, 4))
    >>> r1 = matvec(a, b)
    >>> r2 = np.array(list(map(np.dot, a, b)))
    >>> np.allclose(r1, r2)
    True
    """
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError("vertices and matrices should have same dimension")
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
        raise ValueError("matrices and vectors must be compatible for matrix-vector multiplication")
    if matrices.shape[:-2] != vectors.shape[:-1]:
        shp = np.broadcast(matrices[..., 0, 0], vectors[..., 0]).shape
        matrices_bc = np.broadcast_to(matrices, shp + matrices.shape[-2:])
        vectors_bc = np.broadcast_to(vectors, shp + vectors.shape[-1:])
    else:
        matrices_bc, vectors_bc = matrices, vectors
    matrices_contig = matrices_bc.reshape(-1, matrices.shape[-2], matrices.shape[-1])
    vectors_contig = vectors_bc.reshape(-1, vectors.shape[-1])
    return _fastmath_ext.matvec(matrices_contig, vectors_contig).reshape(matrices_bc.shape[:-2] + (matrices.shape[-2], ))
