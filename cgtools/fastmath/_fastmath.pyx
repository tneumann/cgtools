import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def inv3(np.ndarray[np.float64_t, ndim=3] Ts):
    cdef np.ndarray[np.float64_t, ndim=3] Qs = np.empty_like(Ts) # result
    cdef double det
    cdef double invDet
    cdef int i
    cdef int n_matrices = Ts.shape[0]
    cdef double T00
    cdef double T01
    cdef double T02
    cdef double T10
    cdef double T11
    cdef double T12
    cdef double T20
    cdef double T21
    cdef double T22

    for i in range(n_matrices):
        T00 = Ts[i,0,0]
        T01 = Ts[i,0,1]
        T02 = Ts[i,0,2]
        T10 = Ts[i,1,0]
        T11 = Ts[i,1,1]
        T12 = Ts[i,1,2]
        T20 = Ts[i,2,0]
        T21 = Ts[i,2,1]
        T22 = Ts[i,2,2]
        det = T00 * (T22 * T11 - T21 * T12) \
            - T10 * (T22 * T01 - T21 * T02) \
            + T20 * (T12 * T01 - T11 * T02);
        invDet = 1. / det;
        Qs[i, 0, 0] =  (T11 * T22 - T21 * T12) * invDet;
        Qs[i, 1, 0] = -(T10 * T22 - T12 * T20) * invDet;
        Qs[i, 2, 0] =  (T10 * T21 - T20 * T11) * invDet;
        Qs[i, 0, 1] = -(T01 * T22 - T02 * T21) * invDet;
        Qs[i, 1, 1] =  (T00 * T22 - T02 * T20) * invDet;
        Qs[i, 2, 1] = -(T00 * T21 - T20 * T01) * invDet;
        Qs[i, 0, 2] =  (T01 * T12 - T02 * T11) * invDet;
        Qs[i, 1, 2] = -(T00 * T12 - T10 * T02) * invDet;
        Qs[i, 2, 2] =  (T00 * T11 - T10 * T01) * invDet;

    return Qs


