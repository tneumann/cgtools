import numpy as np

_range = range


def soft_histogram(data, nbins, range=None, normalize=True):
    """Return a soft-binned histogram of data consisting of nbins bins, with an optional range.
    
    If the range is None (default), then the range of the min/max of the given data is used as a range.
    When normalize==True (default), then the final histogram is normalized to 1.

    >>> list(soft_histogram([0, 0.5, 1], 2))
    [0.5, 0.5]
    >>> list(soft_histogram([0, 0.5, 1], 2, normalize=False))
    [1.5, 1.5]
    >>> list(soft_histogram([0], 3, range=(0,3), normalize=False))
    [1.0, 0.0, 0.0]
    >>> list(soft_histogram([1], 3, range=(0,3), normalize=False))
    [0.5, 0.5, 0.0]
    >>> list(soft_histogram([0.5], 3, range=(0,3), normalize=False))
    [1.0, 0.0, 0.0]
    >>> list(soft_histogram([1.5], 3, range=(0,3), normalize=False))
    [0.0, 1.0, 0.0]
    >>> list(soft_histogram([3.0], 3, range=(0,3), normalize=False))
    [0.0, 0.0, 1.0]
    """

    data = np.asfarray(data)
    if range is None:
        dmin, dmax = data.min(), data.max()
    else:
        dmin, dmax = range
        if dmin >= dmax:
            raise ValueError("invalid range given(min >= max)")
        in_range = (data >= dmin) & (data <= dmax)
        if not all(in_range):
            #logging.warn("some data values are outside of the given histogram range, ignoring them")
            data = data[in_range]
    assert data.size > 0
    # move data into the range (0.5 ... nbins + 0.5)
    a = ((data - dmin) / (dmax - dmin)) * (nbins) + 0.5
    # determine the 2 bins that the data values fall into and calculate their weight
    upperweight = a - np.floor(a)
    lowerweight = 1 - upperweight
    lowerbin = a.astype(int) - 1
    upperbin = lowerbin + 1
    # build final soft histogram
    h  = np.zeros(nbins)
    h1 = np.bincount(np.maximum(lowerbin, 0), weights=lowerweight)
    h2 = np.bincount(np.minimum(upperbin, nbins-1), weights=upperweight)
    h[:len(h1)] += h1
    h[:len(h2)] += h2[:nbins] # if an item is exactly dmax, 
                              # then it's upperbin will be nbins 
                              # which is out of the histogram 
                              # (but the value will be zero - ignore those)
    if normalize:
        h /= np.sum(h)
    return h


def soft_histogram_dd(samples, nbins, range, normed=False, wrapping=False):
    D = samples.shape[1]
    if not hasattr(wrapping, '__iter__'):
        wrapping = [wrapping] * D
    nbins = np.array(nbins)
    min, max = list(map(np.array, list(zip(*range))))
    # bring the sample range into the range between [0.5 .. nbins + 0.5]
    a_0_n = ((samples - min) / (max - min)) * nbins + 0.5
    # find for each dimension in which lower bin the sample falls, and with which weight
    lowerbin = []
    lowerweight = []
    for dim in _range(D):
        a_dim = a_0_n[:,dim]
        lowerweight.append(1 - a_dim + np.floor(a_dim))
        lowerbin.append(a_dim.astype(int) - 1)
    lowerbin = np.column_stack(lowerbin)
    lowerweight = np.column_stack(lowerweight)
    # use mgrid to generate corners of the D-dimensional cube of length 1, 
    # which are needed as offsets to lowerbin
    cube = np.mgrid[[slice(0, 2, None)] * D].T.reshape((2**D, -1))
    bin = lowerbin[:,np.newaxis, :] + cube[np.newaxis,:,:]
    for dim in _range(D):
        if wrapping[dim]:
            bin[:,:,dim] = bin[:,:,dim] % nbins[dim]
        else:
            bin[:,:,dim] = np.clip(bin[:,:,dim], 0, nbins[dim])
    w = np.abs( cube[np.newaxis,:,:] - lowerweight[:,np.newaxis,:] )
    weight = np.product(w, axis=-1) # for D==2 this is bilinear filtering
    # given the weights and the corresponding bins on the corners of the data ranges,
    # we can simply use the numpy histogramdd function since it supports weighting smaple points
    return np.histogramdd(bin.reshape((-1, D)), weights=weight.ravel(), 
                          range=[(0,n) for n in nbins], bins=nbins, normed=normed)[0]

