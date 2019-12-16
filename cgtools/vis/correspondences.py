import numpy as np
import traits.api as ta
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data, configure_input
import traitsui.api as tu

from .mesh import mesh_as_vtk_actor


def visualize_point_correspondences(source_pts, target_pts, ij_corr=None, scalars=None, point_size=10):
    if ij_corr is None:
        if source_pts.shape != target_pts.shape:
            raise ValueError("must have same amount of source and target points, or specify ij_corr parameter")
        ij_corr = np.column_stack((np.arange(len(source_pts)), np.arange(len(target_pts))))

    p = source_pts[ij_corr[:,0]]
    p2 = target_pts[ij_corr[:,1]]

    pd = tvtk.PolyData(points=p, verts=np.r_[:len(p)].reshape((-1,1)))
    actor = tvtk.Actor(mapper=tvtk.PolyDataMapper())
    configure_input_data(actor.mapper, pd)
    actor.property.point_size = point_size
    if scalars is not None:
        pd.point_data.scalars = scalars
        actor.mapper.scalar_range = scalars.min(), scalars.max()
    mlab.gcf().scene.add_actor(actor)

    class Ctrl(ta.HasTraits):
        alpha = ta.Range(0., 1.)

        def _alpha_changed(self):
            pd.points = p + self.alpha * (p2 - p)
            mlab.gcf().scene.render()

    Ctrl().configure_traits()


class Morpher(ta.HasTraits):
    alpha = ta.Range(0.0, 1.0)

    def __init__(self, verts1, verts2, tris=None, lines=None, as_points=False, 
                 scalars=None, scalars2=None, vmin=None, vmax=None,
                 actor_property=dict(specular=0.1, specular_power=128., diffuse=0.5),
                 ):
        if tris is None:
            if lines is None:
                rep = 'points'
            else:
                rep = 'wireframe'
        else:
            rep = 'surface'
        super(Morpher, self).__init__()
        self._verts1, self._verts2 = verts1, verts2
        self._polydata = tvtk.PolyData(points=verts1)
        if rep == 'points':
            self._polydata.verts = np.r_[:len(verts1)].reshape(-1,1)
        if tris is not None:
            self._polydata.polys = tris
        if lines is not None:
            self._polydata.lines = lines
        n = tvtk.PolyDataNormals(splitting=False)
        configure_input_data(n, self._polydata)
        self._actor = tvtk.Actor(mapper=tvtk.PolyDataMapper())
        configure_input(self._actor.mapper, n)
        self._actor.property.representation = rep
        if rep == 'points':
            self._actor.property.point_size = 5
        if as_points and scalars is None:
            self._polydata.point_data.scalars = \
                    np.random.uniform(0, 255, (len(verts1), 3)).astype(np.uint8)

        self._scalars12 = None
        if scalars is not None:
            self._polydata.point_data.scalars = scalars
            # automatically determine minimum/maximum from scalars if not given by user
            if vmin is None:
                vmin = scalars.min()
                if scalars2 is not None:
                    vmin = min(vmin, scalars2.min())
            if vmax is None:
                vmax = scalars.max()
                if scalars2 is not None:
                    vmax = max(vmax, scalars2.max())
            if scalars.ndim == 1:
                self._actor.mapper.use_lookup_table_scalar_range = False
                self._actor.mapper.scalar_range = (vmin, vmax)
                self._actor.mapper.lookup_table.hue_range = (0.33, 0)
            # when scalars of second mesh given we need to store both scalars in order
            # to interpolate between them during rendering
            if scalars2 is not None:
                self._scalars12 = (scalars, scalars2)
        else:
            self._actor.property.set(**actor_property)
        mlab.gcf().scene.add_actor(self._actor)

    def _alpha_changed(self):
        self._polydata.points = self._verts1 * (1 - self.alpha) \
                              + self._verts2 * self.alpha
        if self._scalars12 is not None:
            blended = self._scalars12[0] * (1 - self.alpha) \
                    + self._scalars12[1] * self.alpha
            # when scalars is a (n_verts, 3) color array (type uint8)
            # then above blending will cast to float, undo this here:
            if self._scalars12[0].dtype == np.uint8:
                blended = blended.astype(np.uint8)
            self._polydata.point_data.scalars = blended
        mlab.gcf().scene.render()

    traits_view = tu.View(tu.Item('alpha', show_label=False), title='cgtools Morpher')


def visualize_mesh_morph(verts1, verts2, tris=None, **kwargs):
    Morpher(verts1, verts2, tris, **kwargs).configure_traits()


morph_mesh = visualize_mesh_morph


class MultiMeshMorpher(ta.HasTraits):
    visible = ta.Enum(values='_names')
    morph_target = ta.Enum(values='_names')
    morph_alpha = ta.Range(0.0, 1.0, 0.0)
    show_edges = ta.Bool(False)
    _names = ta.List()

    def __init__(self, list_verts, tris, names=None, fig=None, **kw):
        super(MultiMeshMorpher, self).__init__(**kw)
        self._list_verts = list_verts
        self._tris = tris

        if fig is None:
            self._fig = mlab.figure(bgcolor=(1, 1, 1))
        else:
            self._fig = fig

        if names is None:
            names = map(str, range(len(list_verts)))
        self._names = list(names)

        self._verts_by_name = dict(zip(self._names, list_verts))
        self._actor, self._pd = mesh_as_vtk_actor(list_verts[0], tris, return_polydata=True)
        self._actor.property.set(
            ambient=0.0,
            specular=0.15, 
            specular_power=128., 
            diffuse=0.8,
        )
        self._fig.scene.add_actor(self._actor)

        self.visible = self._names[0]
        if len(self._names) > 1:
            self.morph_target = self._names[1]

    @ta.on_trait_change('visible, show_edges, morph_target, morph_alpha')
    def _update(self):
        self._actor.property.edge_visibility = self.show_edges
        v1 = self._verts_by_name[self.visible]
        if self.morph_alpha > 0:
            v2 = self._verts_by_name[self.morph_target]
            self._pd.points = v1 * (1 - self.morph_alpha) + v2 * self.morph_alpha
        else:
            self._pd.points = v1
        self._fig.scene.render()

    view = tu.View(
        tu.Group(
            tu.Item('visible'),
            tu.Item('morph_target'),
            tu.Item('morph_alpha'),
            tu.Item('show_edges', name='Wireframe'),
            label="Viewer"),
        title="MultiMeshMorpher"
    )


def morph_multimesh(list_verts, tris, **kw):
    MultiMeshMorpher(list_verts, tris, **kw).configure_traits()
