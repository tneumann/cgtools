from __future__ import print_function

import numpy as np
from cgtools.fastmath import inv3, inv2

if __name__ == '__main__':
    import timeit
    for dim in [2, 3]:
        print(("testing inv%d" % dim))
        a = np.random.random((1000, dim, dim))
        inv_func = inv2 if dim == 2 else inv3
        def t_cgtools():
            inv_func(a)
        def t_np():
            np.array(list(map(np.linalg.inv, a)))
        speed1 = np.mean(timeit.repeat(t_cgtools, repeat=5, number=100))
        print(("  cgtools: %f" % speed1))
        speed2 = np.mean(timeit.repeat(t_np, repeat=5, number=100))
        print(("  numpy: %f" % speed2))
        print(("  speedup %.2f" % np.mean(speed2 / speed1)))
