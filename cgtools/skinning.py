import numpy as np
import vector as V


def rbm_to_dualquat(rbm):
    import cgkit.cgtypes as cg
    q0 = cg.quat().fromMat(cg.mat3(rbm[:3,:3].T.tolist()))
    q0 = q0.normalize()
    q0 = np.array([q0.w, q0.x, q0.y, q0.z])
    t = rbm[:3, 3]
    q1 = np.array([
            -0.5*( t[0]*q0[1] + t[1]*q0[2] + t[2]*q0[3]),
             0.5*( t[0]*q0[0] + t[1]*q0[3] - t[2]*q0[2]),
             0.5*(-t[0]*q0[3] + t[1]*q0[0] + t[2]*q0[1]),
             0.5*( t[0]*q0[2] - t[1]*q0[1] + t[2]*q0[0]) ])
    return np.array(q0.tolist() + q1.tolist())

def dualquats_to_rbms(blendq):
    qn = blendq[:,:4]
    qd = blendq[:,4:]
    len2 = np.sum(qn**2, axis=1)
    w, x, y, z = qn[:,0], qn[:,1], qn[:,2], qn[:,3]
    t0, t1, t2, t3 = qd[:,0], qd[:,1], qd[:,2], qd[:,3]
    M = np.empty((len(blendq), 4, 4))
    M[:,0,0] = w*w + x*x - y*y - z*z
    M[:,0,1] = 2*x*y - 2*w*z
    M[:,0,2] = 2*x*z + 2*w*y
    M[:,1,0] = 2*x*y + 2*w*z
    M[:,1,1] = w*w + y*y - x*x - z*z
    M[:,1,2] = 2*y*z - 2*w*x;
    M[:,2,0] = 2*x*z - 2*w*y
    M[:,2,1] = 2*y*z + 2*w*x
    M[:,2,2] = w*w + z*z - x*x - y*y
    M[:,0,3] = -2*t0*x + 2*w*t1 - 2*t2*z + 2*y*t3
    M[:,1,3] = -2*t0*y + 2*t1*z - 2*x*t3 + 2*w*t2
    M[:,2,3] = -2*t0*z + 2*x*t2 + 2*w*t3 - 2*t1*y
    M[:,3] = 0
    M[:,3,3] = len2
    M /= len2[:,np.newaxis,np.newaxis]
    return M

def dq_skinning(pts, BW, dqs):
    from scipy import weave

    blendq = np.sum(BW[:,:,np.newaxis] * dqs[np.newaxis], axis=1)
    code = """
    using namespace blitz;
    float M00, M01, M02, M03;
    float M10, M11, M12, M13;
    float M20, M21, M22, M23;
    
    for (int i=0; i<num_pts; i++) {
        float w = blendq(i,0);
        float x = blendq(i,1);
        float y = blendq(i,2);
        float z = blendq(i,3);
        float t0 = blendq(i,4);
        float t1 = blendq(i,5);
        float t2 = blendq(i,6);
        float t3 = blendq(i,7);
        float len2 = 1. / (w*w + x*x + y*y + z*z);
        M00 = (w*w + x*x - y*y - z*z) * len2;
        M01 = (2*x*y - 2*w*z) * len2;
        M02 = (2*x*z + 2*w*y) * len2;
        M10 = (2*x*y + 2*w*z) * len2;
        M11 = (w*w + y*y - x*x - z*z) * len2;
        M12 = (2*y*z - 2*w*x) * len2;
        M20 = (2*x*z - 2*w*y) * len2;
        M21 = (2*y*z + 2*w*x) * len2;
        M22 = (w*w + z*z - x*x - y*y) * len2;
        M03 = (-2*t0*x + 2*w*t1 - 2*t2*z + 2*y*t3) * len2;
        M13 = (-2*t0*y + 2*t1*z - 2*x*t3 + 2*w*t2) * len2;
        M23 = (-2*t0*z + 2*x*t2 + 2*w*t3 - 2*t1*y) * len2;
        pts_transformed(i,0) = M00 * pts(i,0) + M01 * pts(i,1) + M02 * pts(i,2) + M03;
        pts_transformed(i,1) = M10 * pts(i,0) + M11 * pts(i,1) + M12 * pts(i,2) + M13;
        pts_transformed(i,2) = M20 * pts(i,0) + M21 * pts(i,1) + M22 * pts(i,2) + M23;
    }
    """
    pts_transformed = np.empty_like(pts)
    num_pts = len(blendq)
    num_bws = BW.shape[1]
    weave.inline(code,
                 ["num_pts", "num_bws", "blendq", "pts_transformed", "pts", "BW"],
                 type_converters=weave.converters.blitz)
    return pts_transformed

def dq_skinning_py(pts, BW, dqs, inverse=False):
    # blend in dual quaternion space
    blendq = np.sum(BW[:,:,np.newaxis] * dqs[np.newaxis], axis=1)
    # convert them back to rigid body motion (4x4)
    M = dualquats_to_rbms(blendq)
    if inverse == True:
        print M
        M = np.array(map(np.linalg.inv, M))
    # transform points with final matrix
    return V.dehom( np.sum(M * V.hom(pts)[:,np.newaxis,:], axis=2) )

def blend_skinning(pts, BW, rbms, method='lbs'):
    """ 
    perform blend skinning of pts given blend weights BW and the 4x4 rigid body motions in rbms
        pts should be an array of points, so the shape should be (num_points, 3)
        BW should be an array of blendweights, so the shape should be (num_points, num_rbms)
        where num_rbms give the number of rigid body motion parts (joints)
        rbms should be an array of shape (num_rbms, 4, 4) - one rigid body motions for each column in BW

        supported methods are "lbs" (linear blend skinning)
        and "dq" (dual quaternion skinning)
    """
    # TODO use masked arrays to accellerate?
    if method == 'lbs':
        transformed_pts = np.tensordot(V.hom(pts), rbms, axes=(1, 2))
        if transformed_pts.shape[-1] == 4:
            transformed_pts = V.dehom(transformed_pts)
        return np.sum(BW[:,:,np.newaxis] * transformed_pts, axis=1)
    elif method == 'dq':
        rbms = np.asanyarray(rbms)
        dqs = np.array(map(rbm_to_dualquat, rbms))
        return dq_skinning(pts, BW, dqs)
    else:
        raise ValueError, "Unknown skinning method"

