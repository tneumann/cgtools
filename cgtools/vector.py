import numpy as np

__all__ = [
    'normalized', 'veclen', 'sq_veclen', 'scaled', 'dot', 'project',
    'homogenize', 'dehomogenize', 'hom', 'dehom', 'ensure_dim', 'hom3', 'hom4',
    'transform',
]

ARR = np.asanyarray

def normalized(vectors):
    """ normalize vector(s) such that ||vectors|| == 1 """
    vectors = ARR(vectors)
    lengths = veclen(vectors)
    return vectors / lengths[..., np.newaxis]

def veclen(vectors):
    """ calculate euclidean norm of a vector or an array of vectors 
        when an nd-array is given, then the norm is computed about the last dimension
    """
    return np.sqrt(np.sum(ARR(vectors)**2, axis=-1))

def sq_veclen(vectors):
    """ calculate squared euclidean norm of a vector or an array of vectors 
        when an nd-array is given, then the norm is computed about the last dimension
    """
    vectors = ARR(vectors)
    return np.sum(vectors**2, axis=vectors.ndim-1)

def scaled(vectors, scale):
    """ scales vectors such that ||vectors|| == scale """
    return normalized(vectors) * ARR(scale)[..., np.newaxis]

def dot(v, u):
    """ pairwise dot product of 2 vector arrays, 
    essentially the same as [sum(vv*uu) for vv,uu in zip(v,u)], 
    that is, pairwise application of the inner product
    >>> dot([1, 0], [0, 1])
    0
    >>> dot([1, 1], [2, 3])
    5
    >>> dot([[1, 0], [1, 1]], [[0, 1], [2, 3]]).tolist()
    [0, 5]
    """
    return np.sum(ARR(u)*ARR(v), axis=-1)

def project(v, u):
    """ project v onto u """
    u_norm = normalized(u)
    return (dot(v, u_norm)[..., np.newaxis] * u_norm)

def homogenize(v, value=1):
    """ returns v as homogeneous vectors by inserting one more element into the last axis 
    the parameter value defines which value to insert (meaningful values would be 0 and 1) 
    >>> homogenize([1, 2, 3]).tolist()
    [1, 2, 3, 1]
    >>> homogenize([1, 2, 3], 9).tolist()
    [1, 2, 3, 9]
    >>> homogenize([[1, 2], [3, 4]]).tolist()
    [[1, 2, 1], [3, 4, 1]]
    >>> homogenize([[1, 2], [3, 4]], 99).tolist()
    [[1, 2, 99], [3, 4, 99]]
    >>> homogenize([[1, 2], [3, 4]], [33, 99]).tolist()
    [[1, 2, 33], [3, 4, 99]]
    """
    v = ARR(v)
    if hasattr(value, '__len__'):
        return np.append(v, ARR(value).reshape(v.shape[:-1] + (1,)), axis=-1)
    else:
        return np.insert(v, v.shape[-1], np.array(value, v.dtype), axis=-1)

# just some handy aliases
hom = homogenize

def dehomogenize(a):
    """ makes homogeneous vectors inhomogenious by dividing by the last element in the last axis 
    >>> dehomogenize([1, 2, 4, 2]).tolist()
    [0.5, 1.0, 2.0]
    >>> dehomogenize([[1, 2], [4, 4]]).tolist()
    [[0.5], [1.0]]
    """
    a = np.asfarray(a)
    return a[...,:-1] / a[...,np.newaxis,-1]

# just some handy aliases
dehom = dehomogenize

def ensure_dim(a, dim, value=1):
    """
    checks if an array of vectors has dimension dim, and if not,
    adds one dimension with values set to value (default 1)
    """
    cdim = a.shape[-1]
    if cdim == dim - 1:
        return homogenize(a, value=value)
    elif cdim == dim:
        return a
    else:
        raise ValueError('vectors had %d dimensions, but expected %d or %d' % (cdim, dim-1, dim))

def hom4(a, value=1):
    return ensure_dim(a, 4, value)

def hom3(a, value=1):
    return ensure_dim(a, 3, value)

def transform(v, M, w=1):
    """
    transforms vectors in v with the matrix M
    if matrix M has one more dimension then the vectors
    this will be done by homogenizing the vectors
    (with the last dimension filled with w) and
    then applying the transformation

    TODO: unit tests
    """
    if M.shape[0] == M.shape[1] == v.shape[-1] + 1:
        v1 = hom(v, value=w)
        return dehom(np.dot(v1.reshape((-1,v1.shape[-1])), M.T)).reshape(v.shape)
    else:
        return np.dot(v.reshape((-1,v.shape[-1])), M.T).reshape(v.shape)

def toskewsym(v):
    assert v.shape == (3,)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

