import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input, configure_input_data



def mesh_as_vtk_actor(verts, tris=None, lines=None, compute_normals=True, return_polydata=True, scalars=None):
    pd = tvtk.PolyData(points=verts)
    if tris is not None:
        pd.polys = tris
    if lines is not None:
        pd.lines = lines
    if scalars is not None:
        pd.point_data.scalars = scalars
    actor = polydata_actor(pd, compute_normals=compute_normals)
    if return_polydata:
        return actor, pd
    else:
        return actor

def polydata_actor(polydata, compute_normals=True):
    """ create a vtk actor with given polydata as input """
    if compute_normals:
        normals = tvtk.PolyDataNormals(splitting=False)
        configure_input_data(normals, polydata)
        polydata = normals
    actor = tvtk.Actor(mapper=tvtk.PolyDataMapper())
    configure_input(actor.mapper, polydata)
    return actor


def vismesh(pts, tris, color=None, edge_visibility=False, shader=None, triangle_scalars=None, colors=None, **kwargs):
    if 'scalars' in kwargs and np.asarray(kwargs['scalars']).ndim == 2:
        colors = kwargs['scalars']
        del kwargs['scalars']
    # VTK does not allow bool arrays as scalars normally, so convert to float 
    if 'scalars' in kwargs and np.asarray(kwargs['scalars']).dtype == np.bool:
        kwargs['scalars'] = kwargs['scalars'].astype(np.float)

    tm = mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, color=color, **kwargs)
    if shader is not None:
        tm.actor.property.load_material(shader)
        tm.actor.actor.property.shading = True
    diffuse = 1.0 if colors is not None else 0.8
    tm.actor.actor.property.set(
        edge_visibility=edge_visibility, line_width=1, 
        specular=0.0, specular_power=128., 
        diffuse=diffuse)
    if triangle_scalars is not None:
        tm.mlab_source.dataset.cell_data.scalars = triangle_scalars
        tm.actor.mapper.set(scalar_mode='use_cell_data', use_lookup_table_scalar_range=False,
                            scalar_visibility=True)
        if "vmin" in kwargs and "vmax" in kwargs:
            tm.actor.mapper.scalar_range = kwargs["vmin"], kwargs["vmax"]
    if colors is not None:
        # this basically is a hack which doesn't quite work, 
        # we have to completely replace the polydata behind the hands of mayavi
        tm.mlab_source.dataset.point_data.scalars = colors.astype(np.uint8)
        tm.actor.mapper.input = tvtk.PolyDataNormals(
            input=tm.mlab_source.dataset, splitting=False).output
    return tm

def compute_normals(pts, faces):
    pd = tvtk.PolyData(points=pts, polys=faces)
    n = tvtk.PolyDataNormals(splitting=False)
    configure_input_data(n, pd)
    n.update()
    return n.output.point_data.normals.to_array()

def showmesh(pts, tris, **kwargs):
    mlab.clf()
    vismesh(pts, tris, **kwargs)
    if 'scalars' in kwargs:
        mlab.colorbar()
    mlab.show()

def viscroud(meshes, axis=0, padding=1.2, **kwargs):
    offset = np.zeros(3)
    offset[axis] = np.max([mesh[0][:,axis].ptp() for mesh in meshes]) * padding
    tms = []
    # find common minimum when multiple scalars are given per mesh
    # find common minimum when multiple scalars are given per mesh
    scalars = [mesh[2] for mesh in meshes if len(mesh) > 2]
    if len(scalars) > 0:
        kwargs['vmin'] = np.min(scalars)
        kwargs['vmax'] = np.max(scalars)

    for i, mesh in enumerate(meshes):
        verts, tris = mesh[:2]
        scalars = mesh[2] if len(mesh) > 2 else None
        tm = vismesh(verts + offset * i, tris, scalars=scalars, **kwargs)
        tms.append(tm)
    return tms


