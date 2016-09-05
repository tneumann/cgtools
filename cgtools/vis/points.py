import numpy as np
from mayavi import mlab
from tvtk.api import tvtk


def pointcloud_as_vtk_polydata(points, scalars=None, pt_colors=None, polydata=None):
    """ generate tvtk.PolyData (or fill existing polydata) to be displayed as a point cloud """
    if polydata is None:
        polydata = tvtk.PolyData()
    polydata.set(points=points, verts=np.r_[:len(points)].reshape((-1,1)))
    if scalars is not None:
        polydata.point_data.scalars = scalars
    if pt_colors is not None:
        polydata.point_data.scalars = pt_colors.astype(np.uint8)
    return polydata

def vispoints(pts, point_colors=None, point_size=5., mode='2dvertex', **kwargs):
    v = mlab.points3d(
        pts[:,0], pts[:,1], pts[:,2], 
        scale_mode='none', scale_factor=point_size, 
        mode=mode, **kwargs)
    if mode == '2dvertex':
        v.actor.property.point_size = point_size
    if point_colors != None:
        v.glyph.glyph.input.point_data.scalars = point_colors.astype(np.uint8)
    return v

