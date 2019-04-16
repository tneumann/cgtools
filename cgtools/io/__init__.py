from functools import partial
from os import path

from .obj import load_obj, save_obj
from .off import load_off, save_off


def load_mesh(filename):
    loaders = {
        'obj': load_obj,
        'off': partial(load_off, no_colors=True),
    }
    ext = path.splitext(filename)[1].lower()[1:]
    if ext not in loaders:
        raise IOError("No loader for %s extension known, available file formats are: %s" % (ext, list(loaders.keys())))
    return loaders[ext](filename)


def save_mesh(filename, verts, tris, *args, **kw):
    writers = {
        'obj': save_obj,
        'off': save_off,
    }
    ext = path.splitext(filename)[1].lower()[1:]
    if ext not in writers:
        raise IOError("No known writer for %s extension known, available file formats are: %s" % (ext, list(loaders.keys())))
    return writers[ext](filename, verts, tris, *args, **kw)
