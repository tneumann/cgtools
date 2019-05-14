import numpy as np

from . import vector as V


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
    R = procrustes(frompts_local, topts_local, allow_reflection=allow_reflection)
    T0 = np.eye(4)
    T0[:3,:3] = R
    T0[:3, 3] = t1 - np.dot(R, t0)
    return T0


def procrustes(frompts, topts, allow_reflection=False):
    """
    Finds a orthogonal rotation R \in SO(N) 
    that aligns array of N-dimensional frompts with topts.

    Solves the orthogonal procrustes problem.
    """
    M = np.dot(topts.T, frompts)
    U, s, Vt = np.linalg.svd(M)
    if allow_reflection:
        R = U.dot(Vt)
    else:
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        E = np.diag([1, 1, d])
        R = np.dot(U, np.dot(E, Vt))
    return R


def rigid_align(frompts, topts, allow_reflection=False):
    M = procrustes3d(frompts, topts, allow_reflection=allow_reflection)
    return V.transform(frompts, M)