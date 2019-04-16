from __future__ import print_function

import numpy as np
from cgtools.fastmath import cross3


if __name__ == '__main__':
    import timeit
    a = np.random.random((17000, 3))
    b = np.random.random((17000, 3))
    np_cross = np.cross
    def t_cross3():
        cross3(a, b)
    def t_cross_np():
        np_cross(a, b)
    print("blitzcross:", end=' ') 
    speed1 = np.mean(timeit.repeat(t_cross3, repeat=5, number=100))
    print(speed1)
    print("numpy.cross:", end=' ') 
    speed2 = np.mean(timeit.repeat(t_cross_np, repeat=5, number=100))
    print(speed2)
    print("speedup %.2f" % np.mean(speed2 / speed1))
