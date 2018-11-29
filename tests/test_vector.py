import numpy as np
import numpy.testing as npt
from cgtools import vector as V


def test_inv_3x4():
    Xs = np.random.random((1000, 3, 4))
    Xs_inv_np = np.array([np.linalg.inv(V.to_4x4(Xi)) for Xi in Xs])
    Xs_inv_ours = V.inv_3x4(Xs)
    assert np.allclose(Xs_inv_np[:, 3, :], np.array([0, 0, 0, 1])[np.newaxis])
    npt.assert_allclose(Xs_inv_ours, Xs_inv_np[:, :3, :])


def test_transform_many_matrices_many_vectors():
    for dim in [2, 3, 4]:
        # no translation:
        vectors = np.random.random((1000, dim))
        xforms = np.random.random(((1000, dim, dim)))
        ref = np.array([
            M.dot(v)
            for v, M in zip(vectors, xforms)
        ])
        npt.assert_allclose(ref, V.transform(vectors, xforms))

        # with translation, no perspective (e.g. the common 3x4 matrices)
        vectors = np.random.random((1000, dim))
        xforms = np.random.random(((1000, dim, dim + 1)))
        ref = np.array([
            V.dehom(M.dot(V.hom(v)))
            for v, M in zip(vectors, xforms)
        ])
        npt.assert_allclose(ref, V.transform(vectors, xforms))

        # with translation, no perspective
        vectors = np.random.random((1000, dim))
        xforms = np.random.random(((1000, dim + 1, dim + 1)))
        ref = np.array([
            V.dehom(M.dot(V.hom(v)))
            for v, M in zip(vectors, xforms)
        ])
        npt.assert_allclose(ref, V.transform(vectors, xforms))

def test_transform_one_matrix_many_vectors():
    for dim in [2, 3, 4]:
        # no translation:
        vectors = np.random.random((1000, dim))
        M = np.random.random(((dim, dim)))
        ref = np.array([M.dot(v) for v in vectors])
        npt.assert_allclose(ref, V.transform(vectors, M))

        # with translation, no perspective (e.g. the common 3x4 matrices)
        vectors = np.random.random((1000, dim))
        M = np.random.random(((dim, dim + 1)))
        ref = np.array([V.dehom(M.dot(V.hom(v))) for v in vectors])
        npt.assert_allclose(ref, V.transform(vectors, M))

        # with translation, no perspective
        vectors = np.random.random((1000, dim))
        M = np.random.random(((dim + 1, dim + 1)))
        ref = np.array([V.dehom(M.dot(V.hom(v))) for v in vectors])
        npt.assert_allclose(ref, V.transform(vectors, M))

def test_transform_many_matrices_one_vector():
    for dim in [2, 3, 4]:
        # no translation:
        v = np.random.random((dim))
        xforms = np.random.random(((1000, dim, dim)))
        ref = np.array([ M.dot(v) for M in xforms ])
        npt.assert_allclose(ref, V.transform(v, xforms))

        # with translation, no perspective (e.g. the common 3x4 matrices)
        v = np.random.random((dim))
        xforms = np.random.random(((1000, dim, dim + 1)))
        ref = np.array([ V.dehom(M.dot(V.hom(v))) for M in xforms ])
        npt.assert_allclose(ref, V.transform(v, xforms))

        # with translation, no perspective
        v = np.random.random((dim))
        xforms = np.random.random(((1000, dim + 1, dim + 1)))
        ref = np.array([ V.dehom(M.dot(V.hom(v))) for M in xforms ])
        npt.assert_allclose(ref, V.transform(v, xforms))

def test_transform_one_matrices_one_vector():
    for dim in [2, 3, 4]:
        # no translation:
        v = np.random.random((dim))
        M = np.random.random(((dim, dim)))
        ref = M.dot(v)
        npt.assert_allclose(ref, V.transform(v, M))

        # with translation, no perspective (e.g. the common 3x4 matrices)
        v = np.random.random((dim))
        M = np.random.random(((dim, dim + 1)))
        ref = V.dehom(M.dot(V.hom(v)))
        npt.assert_allclose(ref, V.transform(v, M))

        # with translation, no perspective
        v = np.random.random((dim))
        M = np.random.random(((dim + 1, dim + 1)))
        ref = V.dehom(M.dot(V.hom(v)))
        npt.assert_allclose(ref, V.transform(v, M))