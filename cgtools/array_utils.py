import numpy as np

def concatenate_fill(arrays, axis=0, fill_value=None):
    """
    Appends to all the arrays so that they can be concatenated along the given axis (kwargs axis=0 by default).
    The fill_value will be automatically determined from the dtype of arrays. For floating point types of arrays
    it will be set to NaN, for integer arrays it will be either 0 (for unsigned int) or -1  (signed int).

    >>> a = np.arange(2*3).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> b = np.arange(2*2).reshape(2, 2)
    >>> b
    array([[0, 1],
           [2, 3]])
    >>> np.concatenate((a, b), axis=0)
    Traceback (most recent call last):
    ...
    ValueError: all the input array dimensions except for the concatenation axis must match exactly

    >>> concatenate_fill((a, b), axis=0, fill_value=9)
    array([[0, 1, 2],
           [3, 4, 5],
           [0, 1, 9],
           [2, 3, 9]])

    """
    if len(arrays) == 0:
        raise ValueError("Need at least one array")
    if len(arrays) == 1:
        return arrays[0]
    if not all(a.ndim == arrays[0].ndim for a in arrays):
        raise ValueError("Requires arrays with the same number of dimensions")
    if len(set(a.shape for a in arrays)) == 1:
        # all arrays have the same shape, can use normal concatenate
        return np.concatenate(arrays, axis=axis)
    if all(a.shape[axis] == 0 for a in arrays):
        # all arrays are empty along the shape that we want them to be concatenated
        # in this case just return the first array (it is empty anyways)
        return arrays[0]

    final_shape = [(sum if ax == axis else max)(a.shape[ax] for a in arrays)
                   for ax in xrange(arrays[0].ndim)]
    final_dtype = np.result_type(*arrays)
    if fill_value is None:
        if issubclass(final_dtype.type, np.floating):
            fill_value = np.nan
        elif issubclass(final_dtype.type, np.integer):
            fill_value = max(-1, np.iinfo(final_dtype).min)
        else:
            raise ValueError("cannot automatically decide for a fill_value for dtype=%s, please specify fill_value explicitely" % str(final_dtype))

    concat = np.full(final_shape, fill_value, dtype=final_dtype)
    i = 0
    for a in arrays:
        target = [slice(0, a.shape[ax], 1) for ax in range(a.ndim)]
        target[axis] = slice(i, i + a.shape[axis], 1)
        concat[target] = a
        i += a.shape[axis]

    return concat

