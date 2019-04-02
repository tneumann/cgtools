import numpy as np
from scipy import sparse
from collections import defaultdict, OrderedDict
from itertools import izip, count


def inverse_index_dict(_list):
    """ build a dict that maps items to their index in given _list

    >>> inverse_index_dict(['zero', 'one', 'two'])
    {'zero': 0, 'two': 2, 'one': 1}

    be careful with collisions, as they are not handled correctly (TODO?)
    >>> inverse_index_dict([55, 66, 77, 55])
    {66: 1, 77: 2, 55: 3}
    
    """
    return dict(zip(_list, range(len(_list))))

def occurance_mask(c1, c2):
    c1_set = set(c1)
    return np.array([c2_item in c1_set for c2_item in c2], np.bool)

def group_all(query, data=None):
    """
    Build a dictionary that allows reverse lookup of elements in data that have same query.
    A dict is build from query, where each unique element in query will be used as a key. 
    The corresponding value will be a list of all data[i] where query[i] == key.
    Query and data should be iterables of equal length containing the corresponding query -> data relations.
    If data==None (default), an index mapping will be built, that is values in the returned dict will
    be lists of indices into query.

    The difference to the itertools.groupby function is that not only adjacent equal elements are grouped,
    but all equal elements in query. 

    >>> group_all([1, 2, 1, 2, 3], ['a', 'b', 'c', 'd', 'f'])
    {1: ['a', 'c'], 2: ['b', 'd'], 3: ['f']}
    >>> group_all([1, 2, 1, 2, 3])
    {1: [0, 2], 2: [1, 3], 3: [4]}
    >>> group_all("hello")
    {'h': [0], 'e': [1], 'l': [2, 3], 'o': [4]}
    """
    if data is None:
        data = count()
    result = defaultdict(list)
    for q, d in izip(query, data):
        result[q].append(d)
    return dict(result)

def group_all_keep_order(query, data=None):
    """
    Same as group_all, but returns an OrderedDict, keys are kept in the same order as they appear in query

    >>> group_all_keep_order("haha")
    OrderedDict([('h', [0, 2]), ('a', [1, 3])])
    >>> group_all_keep_order("ahh")
    OrderedDict([('a', [0]), ('h', [1, 2])])
    """
    if data is None:
        data = count()
    result = OrderedDict()
    for q, d in izip(query, data):
        if q not in result:
            result[q] = []
        result[q].append(d)
    return result

def group_all_array(query):
    """
    Numpy-optimized version of group_all, which returns a dictionary of (value -> array indices).

    >>> group_all_array([1, 2, 1, 2, 3])
    {1: array([0, 2]), 2: array([1, 3]), 3: array([4])}
    >>> type(group_all_array([1, 2, 1, 2, 3])[1])
    <type 'numpy.ndarray'>
    """
    values, inverse = np.unique(query, return_inverse=True)
    return {v: np.flatnonzero(inverse == i) for i, v in enumerate(values)}
    

def filter_reindex(condition, indices):
    """
    Filtering of index arrays. Filters and reindex a given index array ("indices"). 
    Say you have target[indices] and now you filter target by some condition, e.g.
    target[condition]. Then, target[condition][indices] will not work anymore.
    Use filter_reindex to fix that: target[condition][filter_reindex(condition, indices)]

    To explain this better, let us look at an example. Let's say you have the following data:
    >>> data = np.array(['a', 'b', 'c', 'd'])
    
    You also have another array that consists of indices into this array
    >>> indices = np.array([0, 3, 2, 0])
    >>> print(data[indices])
    ['a' 'd' 'c' 'a']

    Now, say you are filtering some elements in your data array
    >>> condition = (data == 'a') | (data == 'd')
    >>> filtered_data = data[condition]
    >>> print(filtered_data)
    ['a' 'd']

    The problem is that your index array doesn't correctly reference the new filtered data array
    >>> filtered_data[indices]
    Traceback (most recent call last):
        ...
    IndexError: index 3 is out of bounds for axis 1 with size 2

    Based on an old index array, this method returns a new index array 
    that re-indices into the data array as if condition was applied to this array, so
    >>> filtered_indices = filter_reindex(condition, indices)
    >>> print(filtered_indices)
    [0 1 0]
    >>> print(filtered_data[filtered_indices])
    ['a' 'd' 'a']

    >>> indices = np.array([1, 4, 1, 4])
    >>> condition = np.array([False, True, False, False, True])
    >>> print(filter_reindex(condition, indices))
    [0 1 0 1]

    Also works when one of the filtered elements actually is not in the indices
    >>> data = np.array(['a', 'b', 'c'])
    >>> indices = np.array([0, 0, 1, 1])
    >>> condition = np.array([True, False, True])
    >>> print(data[indices])
    ['a' 'a' 'b' 'b']
    >>> indices_filtered = filter_reindex(condition, indices)
    >>> print(indices_filtered)
    [0 0]

    Filtering everything returns an empty index array
    >>> indices = np.array([1, 1, 2, 3])
    >>> condition = np.array([True, False, False, False])
    >>> print(filter_reindex(condition, indices))
    []

    Also works for n-dim arrays, for example to filter a mesh. 
    E.g. given a list of vertices in 2D.
    >>> verts = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)])

    And a list of triangles:
    >>> triangles = np.array([(0, 1, 2), (0, 2, 3), (3, 2, 4)])

    Lets say we want to filter some of the vertices:
    >>> vert_mask = [True, True, True, False, True]
    >>> verts_new = verts[vert_mask]

    Now, if we filter the vertices and keep using the triangles, we will get an index error:
    >>> verts_new[triangles]
    Traceback (most recent call last):
        ...
    IndexError: ...

    This can be fixed by filter_reindex:
    >>> print(filter_reindex(vert_mask, triangles))
    [[0 1 2]]

    This works with n-d indices:
    >>> target = np.random.random((100, 6))
    >>> indices = np.random.randint(len(target), size=(3000, 2, 4, 3))
    >>> target[indices].shape
    (3000, 2, 4, 3, 6)
    >>> mask = (target > 0.8).any(axis=1)
    >>> target_masked = target[mask]
    >>> target_masked[indices]
    Traceback (most recent call last):
        ...
    IndexError: ...
    >>> indices_masked = filter_reindex(mask, indices)
    >>> target_masked[indices_masked].shape[1:]
    (2, 4, 3, 6)

    """
    condition = np.asarray(condition)
    if condition.dtype != np.bool:
        raise ValueError("condition must be a binary array")
    if condition.ndim > 1:
        raise ValueError("condition must be a 1D array")

    # build the reindexing array
    reindex = np.cumsum(condition) - 1

    # filter indices
    if indices.ndim == 1:
        ind_mask = condition[indices]

    else:
        ind_2d = indices.reshape(indices.shape[0], -1)
        # accept only indices that have no items filtered
        ind_mask = condition[ind_2d].all(axis=-1)

    return reindex[indices][ind_mask]


def valid_indices(indices, array_shape, return_mask=False):
    """
    Returns only the valid indices that fall into array_shape.
    If indices is a float array, then the indices will be rounded and converted to integer.

    >>> idx = np.array([(1, 0), (-1, 2), (-3, 5), (0, 0), (6, 7)])
    >>> valid_indices(idx, (2, 2))
    array([[1, 0],
           [0, 0]])

    Can also return the mask which indices would be selected:

    >>> print(valid_indices(idx, (2, 2), return_mask=True)[1])
    [ True False False  True False]

    >>> valid_indices(np.random.randint(0, 2, (10, 3, 2)), 2)
    Traceback (most recent call last):
    ...
    ValueError: indices array must be of shape (?, 1) or must be one-dimensional

    """
    if len(indices) == 0:
        return np.zeros_like(indices, np.int)

    indices = np.asarray(indices)
    if type(array_shape) == int:
        array_shape = (array_shape,)
    if (indices.ndim != 2 and len(array_shape) > 1):
        raise ValueError("indices array must be of shape (?, 2)")
    if (indices.ndim > 2 and len(array_shape) == 1):
        raise ValueError("indices array must be of shape (?, 1) or must be one-dimensional")

    if issubclass(indices.dtype.type, np.floating):
        indices = np.round(indices).astype(np.int)

    if not issubclass(indices.dtype.type, np.integer):
        raise ValueError("indices must be integer arrays")

    was_1d = False
    if indices.ndim == 1:
        indices = indices[:, np.newaxis]
        was_1d = True

    mask = np.ones(indices.shape[0], np.bool)
    for axis in range(indices.shape[1]):
        mask &= (indices[:, axis] >= 0) & (indices[:, axis] < array_shape[axis])
    indices_masked = indices[mask]

    if was_1d:
        indices_masked = indices_masked.ravel()

    if return_mask:
        return indices_masked, mask
    else:
        return indices_masked

def mask_from_indices(ix, count=None):
    """
    Given an array of 1d indices, return a mask array that has all those indices
    set to True, and the remaining indices set to Failse.
    The returned mask has size of given count, or if count == None, 
    the length corresponds to the maximum element in ix.

    >>> print(mask_from_indices([1, 3]))
    [False True False True]
    >>> print(mask_from_indices([1, 3], 5))
    [False True False True False]

    """
    return (np.bincount(ix, minlength=count) != 0)


def sparse_indicator_matrix(ci, num_cols, omega=1.):
    """ 
    build a sparse constraint matrix C 
    for each i in ci, a row is made in the matrix, and the i'th entry is set to omega (default=1)
    such a matrix can be easily used in least squares problems, since C*x[ci] == x[ci]
    omega is the value that is placed into the nonzero entries of the matrix
    >>> x = np.array([4, 5, 6, 7, 8], np.float)
    >>> C = sparse_indicator_matrix([1, 2, 4], 5)
    >>> print(C * x)
    [5. 6. 8.]
    """
    ci = np.asanyarray(ci)
    if ci.dtype == np.bool:
        ci = ci.nonzero()[0]
    data = np.ones(len(ci)) * omega
    ij = (np.arange(len(ci)), ci)
    return sparse.csr_matrix((data, ij), shape=(len(ci), num_cols))

