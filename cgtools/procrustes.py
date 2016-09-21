import numpy as np


def procrustes3d(frompts, topts, allow_reflection=False):
    """
    Finds a rigid body transformation M that moves points in frompts to the points in topts
    that is, it finds a rigid body motion [ R | t ] with R \in SO(3)

    This algorithm first approximates the rotation by solving
    the orthogonal procrustes problem.
    """
    # center data
    t0 = frompts.mean(0)
    t1 = topts.mean(0)
    frompts_local = frompts - t0
    topts_local = topts - t1
    # find best rotation - procrustes problem
    M = np.dot(topts_local.T, frompts_local)
    U, s, Vt = np.linalg.svd(M)
    #if np.linalg.det(np.dot(U, Vt)) < 0:
    #    Vt[-1] *= -1
    #R = np.dot(U, Vt)
    if allow_reflection:
        R = np.dot(Vt.T, U.T)
    else:
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        E = np.diag([1, 1, d])
        R = np.dot(Vt.T, np.dot(E, U.T))
    T0 = np.eye(4)
    T0[:3,:3] = R
    T0[:3, 3] = t1 - np.dot(R, t0)
    return T0

