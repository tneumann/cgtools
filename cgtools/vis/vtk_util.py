from tvtk.api import tvtk

def numpy_matrix_to_vtk_matrix(M):
    assert M.shape == (4, 4)
    Mvtk = tvtk.Matrix4x4()
    for i in xrange(4):
        for j in xrange(4):
            Mvtk.set_element(i, j, M[i, j])
    return Mvtk
