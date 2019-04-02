from mayavi import mlab


def vislines(from_points, to_points, **kwargs):
    x, y, z = from_points.T
    u, v, w = (to_points - from_points).T
    kw = dict(scale_factor=1, scale_mode='vector', mode='2ddash', line_width=1)
    kw.update(kwargs)
    quiv = mlab.quiver3d(x, y, z, u, v, w, **kw)
    if 'scalars' in kw:
        quiv.glyph.color_mode = 'color_by_scalar'
    return quiv
