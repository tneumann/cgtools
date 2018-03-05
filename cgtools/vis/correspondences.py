import numpy as np
from traits.api import Range, HasTraits
from mayavi import mlab
from tvtk.api import tvtk
from traitsui.api import View, Item


def visualize_point_correspondences(source_pts, target_pts, ij_corr=None, scalars=None):
    if ij_corr is None:
        if source_pts.shape != target_pts.shape:
            raise ValueError("must have same amount of source and target points, or specify ij_corr parameter")
        ij_corr = np.column_stack((np.arange(len(source_pts)), np.arange(len(target_pts))))

    p = source_pts[ij_corr[:,0]]
    p2 = target_pts[ij_corr[:,1]]

    pd = tvtk.PolyData(points=p, verts=np.r_[:len(p)].reshape((-1,1)))
    #pd = tvtk.PolyData(points=p, polys=np.arange(len(p)).reshape((-1, 3)))
    actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(input=pd))
    actor.property.point_size=10
    if scalars is not None:
        pd.point_data.scalars = scalars
        actor.mapper.scalar_range = scalars.min(), scalars.max()
    mlab.gcf().scene.add_actor(actor)

    class Ctrl(HasTraits):
        alpha = Range(0., 1.)

        def _alpha_changed(self):
            pd.points = p + self.alpha * (p2 - p)
            mlab.gcf().scene.render()

    Ctrl().edit_traits()

    mlab.show()



class Morpher(HasTraits):
    alpha = Range(0.0, 1.0)

    def __init__(self, verts1, verts2, tris=None, as_points=False, scalars=None,
                 actor_property=dict(specular=0.1, specular_power=128., diffuse=0.5),
                 ):
        if tris is None:
            as_points = True
        HasTraits.__init__(self)
        self._verts1, self._verts2 = verts1, verts2
        self._polydata = tvtk.PolyData(points=verts1)
        if as_points:
            self._polydata.verts = np.r_[:len(verts1)].reshape(-1,1)
        else:
            self._polydata.polys = tris
        n = tvtk.PolyDataNormals(splitting=False)
        if hasattr(n, 'set_input_data'):
            n.set_input_data(self._polydata)
            self._actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(input_connection=n.output_port))
        else:
            n.input = self._polydata
            self._actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(input=n.output))
        if as_points:
            self._actor.property.set(representation='points', point_size=5)
        if as_points and scalars is None:
            self._polydata.point_data.scalars = \
                    np.random.uniform(0, 255, (len(verts1), 3)).astype(np.uint8)
        if scalars is not None:
            self._polydata.point_data.scalars = scalars
            self._actor.mapper.scalar_range = (scalars.min(), scalars.max())
            self._actor.mapper.lookup_table.hue_range = (0.33, 0)
        else:
            self._actor.property.set(**actor_property)
        mlab.gcf().scene.add_actor(self._actor)

    def _alpha_changed(self):
        self._polydata.points = self._verts1 * (1 - self.alpha) + self._verts2 * self.alpha
        mlab.gcf().scene.render()

    traits_view = View(Item('alpha', show_label=False), title='cgtools Morpher')


def visualize_mesh_morph(verts1, verts2, tris=None, **kwargs):
    Morpher(verts1, verts2, tris, **kwargs).configure_traits()


if __name__ == "__main__":
    x, y = map(np.ravel, np.mgrid[:10:20j, :10:20j])
    z1 = np.sin(0.5 * x) + np.cos(1.2 * y)
    z2 = -0.5 * np.sin(0.4 * x) + 0.5 * np.cos(1.0 * y)
    ix = np.arange(len(x)).reshape(20, 20)
    quads = np.column_stack(map(np.ravel, [ix[:-1, :-1], 
                                           ix[ 1:, :-1], 
                                           ix[ 1:, 1:], 
                                           ix[:-1, 1:]]))
    tris = quads[:, [0, 1, 2,  2, 3, 0]].reshape(-1, 3)

    visualize_mesh_morph(
        np.column_stack((x, y, z1)), 
        np.column_stack((x, y, z2)), 
        quads,
        actor_property={'edge_visibility': True},
    )
    
