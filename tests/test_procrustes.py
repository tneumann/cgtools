import numpy as np
from cgtools import vector as V
from cgtools.procrustes import procrustes3d


def test_procrustes3d():
    # test with 100 random rotations/translations
    trials = 0
    while trials < 100:
        R = np.linalg.qr(np.random.uniform(-1, 1, size=(3,3)))[0]
        print((np.linalg.det(R)))
        if np.linalg.det(R) < 0:
            continue
        t = np.random.uniform(-2, 2, size=3)
        M = np.eye(4)
        M[:3,:3] = R
        M[:3, 3] = t
        N = np.random.randint(3, 1000)
        frompts = np.random.random((N, 3))
        topts = V.transform(frompts, M)
        M_est = procrustes3d(frompts, topts)
        np.testing.assert_allclose(M, M_est)
        np.testing.assert_allclose(V.transform(frompts, M_est), topts)
        R = M_est[:3, :3]
        trials += 1

def test_procrustes3d_reflection():
    for axis in [0, 1, 2]:
        N = 100
        pts1 = np.random.random((N, 3))
        S = np.eye(3)
        S[axis, axis] = -1
        pts2 = V.transform(pts1, S)
        M = procrustes3d(pts1, pts2)
        R = M[:3, :3]
        np.testing.assert_allclose(np.linalg.det(R), 1)
        assert not np.allclose(R, S)
        assert V.veclen(V.transform(pts1, M) - pts2).sum() > 0

        M2 = procrustes3d(pts1, pts2, allow_reflection=True)
        np.testing.assert_allclose(M2[:3, :3], S, atol=1.e-9)
        np.testing.assert_allclose(V.transform(pts1, M2), pts2)

