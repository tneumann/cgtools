import numpy as np
from cgtools.fastmath import matmat, matvec, inv2, inv3, cross3


def test_matmat_stride0():
    a = np.random.random((10, 3, 5))
    b = np.random.random((5, 3))
    r1 = matmat(a, b)
    r2 = np.array([np.dot(ai, b) for ai in a])
    np.testing.assert_allclose(r1, r2)
    a = np.random.random((3, 5))
    b = np.random.random((12, 5, 3))
    r1 = matmat(a, b)
    r2 = np.array([np.dot(a, bi) for bi in b])
    np.testing.assert_allclose(r1, r2)

def test_matmat_eqshape():
    a = np.random.random((31, 4, 4))
    b = np.random.random((31, 4, 4))
    r1 = matmat(a, b)
    r2 = np.array(map(np.dot, a, b))
    np.testing.assert_allclose(r1, r2)

def test_matmat_ndim():
    a = np.random.random((10, 11, 2, 4))
    b = np.random.random((10, 11, 4, 3))
    r1 = matmat(a, b)
    r2 = np.array(map(np.dot, a.reshape(-1, 2, 4), b.reshape(-1, 4, 3))).reshape(10, 11, 2, 3)
    np.testing.assert_allclose(r1, r2)

def test_matmat_noncontiguous():
    # a non-contiguous
    a = np.random.random((10, 3, 3)).swapaxes(1, 2)
    b = np.random.random((10, 3, 3))
    assert not a.flags.contiguous
    r1 = matmat(a, b)
    r2 = np.array(map(np.dot, a, b))
    np.testing.assert_allclose(r1, r2)
    # b non-contiguous
    a = np.random.random((10, 3, 4))
    b = np.random.random((10, 3, 4)).swapaxes(1, 2)
    assert not b.flags.contiguous
    r1 = matmat(a, b)
    r2 = np.array(map(np.dot, a, b))
    np.testing.assert_allclose(r1, r2)
    # both non-contiguous
    a = np.random.random((10, 3, 5)).swapaxes(1, 2)
    b = np.random.random((10, 6, 3)).swapaxes(1, 2)
    assert not a.flags.contiguous
    assert not b.flags.contiguous
    r1 = matmat(a, b)
    r2 = np.array(map(np.dot, a, b))
    np.testing.assert_allclose(r1, r2)

def test_matvec_stride0():
    a = np.random.random((10, 3, 5))
    b = np.random.random(5)
    r1 = matvec(a, b)
    r2 = np.array([np.dot(ai, b) for ai in a])
    np.testing.assert_allclose(r1, r2)
    a = np.random.random((3, 5))
    b = np.random.random((12, 5))
    r1 = matvec(a, b)
    r2 = np.array([np.dot(a, bi) for bi in b])
    np.testing.assert_allclose(r1, r2)

def test_matvec_eqshape():
    a = np.random.random((31, 5, 5))
    b = np.random.random((31, 5))
    r1 = matvec(a, b)
    r2 = np.array(map(np.dot, a, b))
    np.testing.assert_allclose(r1, r2)

def test_matvec_single():
    a = np.random.random((3, 3))
    b = np.random.random(3)
    r1 = matvec(a, b)
    r2 = np.dot(a, b)
    assert r1.shape == r2.shape
    np.testing.assert_allclose(r1, r2)

def test_matmat_single():
    a = np.random.random((3, 5))
    b = np.random.random((5, 3))
    r1 = matmat(a, b)
    r2 = np.dot(a, b)
    assert r1.shape == r2.shape
    assert r1.dtype == r2.dtype
    np.testing.assert_allclose(r1, r2)

def test_inv3():
    T = np.random.random((3, 3))
    np.testing.assert_allclose(np.linalg.inv(T), inv3(T))

def test_inv3_multiple():
    Ts = np.random.random((154, 7, 3, 3))
    Tinv_np = np.array(map(np.linalg.inv, Ts.reshape((-1, 3, 3)))).reshape(Ts.shape)
    Tinv_blitz = inv3(Ts)
    np.set_printoptions(suppress=True)
    np.testing.assert_allclose(Tinv_np, Tinv_blitz)

def test_inv3_float32():
    Ts = np.random.random((10, 3, 3)).astype(np.float32)
    Tinv_np = np.array(map(np.linalg.inv, Ts.reshape((-1, 3, 3)))).reshape(Ts.shape)
    Tinv_blitz = inv3(Ts)
    assert Tinv_blitz.dtype == np.float32
    np.set_printoptions(suppress=True)
    np.testing.assert_array_almost_equal(Tinv_np, Tinv_blitz, decimal=4)

def test_inv2():
    T = np.random.random((2, 2))
    np.testing.assert_allclose(np.linalg.inv(T), inv2(T))

def test_inv2_multiple():
    Ts = np.random.random((154, 7, 2, 2))
    Tinv_np = np.array(map(np.linalg.inv, Ts.reshape((-1, 2, 2)))).reshape(Ts.shape)
    Tinv_blitz = inv2(Ts)
    np.set_printoptions(suppress=True)
    np.testing.assert_allclose(Tinv_np, Tinv_blitz)

def test_inv2_float32():
    Ts = np.random.random((10, 2, 2)).astype(np.float32)
    Tinv_np = np.array(map(np.linalg.inv, Ts)).reshape(Ts.shape)
    Tinv_blitz = inv2(Ts)
    assert Tinv_blitz.dtype == np.float32
    np.set_printoptions(suppress=True)
    np.testing.assert_array_almost_equal(Tinv_np, Tinv_blitz, decimal=4)

def test_cross3():
    a = np.random.random((1000, 3))
    b = np.random.random((1000, 3))
    c_numpy = np.cross(a, b)
    c_fast = cross3(a, b)
    np.testing.assert_allclose(c_numpy, c_fast)
