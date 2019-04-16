from __future__ import print_function

import numpy as np
from cgtools.fastmath import matmat, matvec


if __name__ == '__main__':
    import timeit
    a = np.random.random((10000, 4, 4))
    b = np.random.random((10000, 4, 4))

    def t_matmat():
        matmat(a, b)
    def t_naive_np():
        np.array(list(map(np.dot, a, b)))
    def t_matmul():
        np.matmul(a, b)

    print("measuring performance of multiplying %d %dx%d matrices" % a.shape)

    timeit_args = dict(repeat=5, number=100)

    speed_matmat = np.mean(timeit.repeat(t_matmat, **timeit_args))
    print("cgtools matmat:", speed_matmat)

    speed_matmul = np.mean(timeit.repeat(t_matmul, **timeit_args))
    print("numpy matmul: %f (speedup %.2f)" % (speed_matmul, speed_matmul / speed_matmat))

    speed_np = np.mean(timeit.repeat(t_naive_np, **timeit_args))
    print("naive numpy: %f (speedup %.2f)" % (speed_np, speed_np / speed_matmat))


    a = np.random.random((10000, 4, 4))
    b = np.random.random((10000, 4))

    def t_matvec():
        return matvec(a, b)
    def t_matvec_naive_np():
        np.array(list(map(np.dot, a, b)))
    def t_matvec_matmul():
        return np.matmul(a, b[:, :, np.newaxis])[:, :, 0]

    print((np.allclose(t_matvec(), t_matvec_matmul())))

    print()
    print("measuring performance of multiplying %d %dx%d matrices with %d %d-dimensional vectors" % tuple(list(a.shape) + list(b.shape)))

    speed_matmat = np.mean(timeit.repeat(t_matvec, **timeit_args))
    print("cgtools matmat:", speed_matmat)

    speed_matmul = np.mean(timeit.repeat(t_matvec_matmul, **timeit_args))
    print("numpy matmul: %f (speedup %.2f)" % (speed_matmul, speed_matmul / speed_matmat))

    speed_np = np.mean(timeit.repeat(t_matvec_naive_np, **timeit_args))
    print("naive numpy: %f (speedup %.2f)" % (speed_np, speed_np / speed_matmat))
