import numpy as np
import numpy.testing as npt
from cgtools.array_utils import concatenate_fill


def test_concatenate_fill_axis0():
    a = np.random.random((10, 4, 5))
    b = np.random.random((5, 2, 5))
    c = np.random.random((5, 1, 4))
    r = concatenate_fill((a, b, c), axis=0)
    npt.assert_equal(r.shape, (20, 4, 5))
    npt.assert_array_equal(r[ 0:10, :4, :5], a)
    npt.assert_array_equal(r[ 0:10, 4:, 5:], np.nan)
    npt.assert_array_equal(r[10:15, :2, :5], b)
    npt.assert_array_equal(r[10:15, 2:, 5:], np.nan)
    npt.assert_array_equal(r[15:20, :1, :4], c)
    npt.assert_array_equal(r[15:20, 1:, 4:], np.nan)

def test_concatenate_fill_axis1():
    a = np.random.random((10, 4, 5))
    b = np.random.random((8, 2, 3))
    r = concatenate_fill((a, b), axis=1)
    npt.assert_equal(r.shape, (10, 6, 5))
    sa = np.s_[  : , :4, :5]
    sb = np.s_[ 0:8, 4:, :3]
    npt.assert_array_equal(r[sa], a)
    npt.assert_array_equal(r[sb], b)
    r[sa] = np.nan
    r[sb] = np.nan
    npt.assert_array_equal(r, np.nan)

def test_concatenate_fill_one_array():
    a = np.random.random((10, 4, 5))
    r = concatenate_fill((a, ), axis=1)
    npt.assert_array_equal(r, a)

