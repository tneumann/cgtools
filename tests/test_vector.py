import numpy as np
import numpy.testing as npt
from cgtools import vector as V


def test_inv_3x4():
    Xs = np.random.random((1000, 3, 4))
    Xs_inv_np = np.array([np.linalg.inv(V.to_4x4(Xi)) for Xi in Xs])
    Xs_inv_ours = V.inv_3x4(Xs)
    assert np.allclose(Xs_inv_np[:, 3, :], np.array([0, 0, 0, 1])[np.newaxis])
    npt.assert_allclose(Xs_inv_ours, Xs_inv_np[:, :3, :])
