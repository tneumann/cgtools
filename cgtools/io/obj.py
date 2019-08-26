import numpy as np
from os import path
import re

from . import _fastobj_ext


# TODO: use \d  in triangle regex instead of [^\/\s]
_triangle_regex = re.compile(r"^f\s+(\d+)\S*\s+(\d+)\S*\s+(\d+)", re.MULTILINE)
_quad_regex     = re.compile(r"^f\s+(\d+)\S*\s+(\d+)\S*\s+(\d+)\S*\s+(\d+)", re.MULTILINE)
_triangle_regex_all = re.compile(r"^f\s+([\d]+)/?([\d]*)/?([\d]*)\s+([\d]+)/?([\d]*)/?([\d]*)\s+([\d]+)/?([\d]*)/?([\d]*)", re.MULTILINE)
_quad_regex_all = re.compile(r"^f\s+(\d+)/?(\d+)?/?(\d+)?\s+(\d+)/?(\d+)?/?(\d+)?\s+(\d+)/?(\d+)?/?(\d+)?\s+(\d+)/?(\d+)?/?(\d+)?", re.MULTILINE)
_normal_regex = re.compile("^vn\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
_texcoord_regex = re.compile("^vt\s+(\S+)\s+(\S+)", re.MULTILINE)
_vertex_regex = re.compile("^v\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
_vertex_regex_with_color = re.compile("^v\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)


def _array_fromregex(buffer, regex, dtype):
    # could use np.fromregex but that has a bug in some versions of numpy
    # that is caused when called with a compiled regex object
    # see https://github.com/numpy/numpy/pull/10501
    seq = regex.findall(buffer)
    return np.array(seq, dtype=dtype)


def load_texture_filename_in_obj(filename):
    tex_file = None
    for line in open(filename):
        if line.startswith('mtllib'):
            mtl_file = path.join(path.dirname(filename), line.strip().split()[1])
            for mtl_line in open(mtl_file):
                if mtl_line.startswith('map_Kd'):
                    tex_file = mtl_line.strip().split()[1]
    return tex_file


def load_obj(filename, load_normals=False, load_texcoords=False, load_texture=False, 
             load_full_face_definitions=False, is_quadmesh='auto', load_colors=False):
    """ load a wavefront obj file
        loads vertices into a (x,y,z) struct array and vertex indices
        into a n x 3 index array 
        only loads obj files vertex positions and also
        only works with triangle or (when is_quadmesh=True) with quad meshes """
    contents = open(filename).read()
    if load_colors:
        data = _array_fromregex(contents, _vertex_regex_with_color, np.float)
        vertices = data[:,:3]
        colors = data[:,3:]
    else:
        vertices = _array_fromregex(contents, _vertex_regex, np.float)

    if load_normals:
        normals = _array_fromregex(contents, _normal_regex, np.float)
    if load_texcoords:
        texcoords = _array_fromregex(contents, _texcoord_regex, np.float)

    reg_quads = _quad_regex_all if load_full_face_definitions else _quad_regex
    reg_tris = _triangle_regex_all if load_full_face_definitions else _triangle_regex
    if is_quadmesh == 'auto':
        quads = _array_fromregex(contents, reg_quads, np.int) - 1 # 1-based indexing in obj file format!
        if len(quads) > 0:
            faces = quads
        else:
            faces = _array_fromregex(contents, reg_tris, np.int) - 1 # 1-based indexing in obj file format!
    elif is_quadmesh:
        quads = _array_fromregex(contents, reg_quads, np.int) - 1 # 1-based indexing in obj file format!
        faces = quads
    else:
        tris = _array_fromregex(contents, reg_tris, np.int) - 1 # 1-based indexing in obj file format!
        faces = tris

    r = [vertices]
    if load_normals:
        r.append(normals)
    if load_texcoords:
        r.append(texcoords)
    if load_colors:
        r.append(colors)

    r.append(faces)

    if load_texture:
        from scipy.misc import imread
        
        tex_file = load_texture_filename_in_obj(filename)
        if tex_file is None:
            raise IOError("Cannot read texture from %s" % filename)
        texture = imread(path.join(path.dirname(filename), tex_file))
        r.append(texture)

    return r


def load_obj_fast(filename, *args, **kw):
    return _fastobj_ext.load_obj_fast(filename)


def save_obj(filename, vertices, faces, normals=None, texcoords=None, texture_file=None):
    with open(filename, 'w') as f:
        if texture_file is not None:
            mtl_file = filename + ".mtl"
            f.write("mtllib ./%s\n\n" % path.basename(mtl_file))
            if path.dirname(texture_file) == path.dirname(filename):
                texture_file = path.basename(texture_file)
            with open(mtl_file, 'w') as mf:
                mf.write(
                    "newmtl material_0\n"
                    "Ka 0.200000 0.200000 0.200000\n"
                    "Kd 1.000000 1.000000 1.000000\n"
                    "Ks 1.000000 1.000000 1.000000\n"
                    "Tr 1.000000\n"
                    "illum 2\n"
                    "Ns 0.000000\n"
                    "map_Kd %s\n" % texture_file
                )
        np.savetxt(f, vertices, fmt="v %f %f %f")
        if texcoords is not None:
            np.savetxt(f, texcoords, 
                       fmt="vt %f %f")

        if normals is not None:
            np.savetxt(f, normals, 
                       fmt="vn %f %f %f")

        if texture_file is not None:
            f.write("usemtl material_0\n\n")

        if faces is not None and len(faces) > 0:
            if faces.shape[1] in [3, 4]:
                face_fmt = 'f ' + ' '.join(['%d'] * faces.shape[1])
            elif faces.shape[1] in [3*3, 3*4]:
                face_fmt = 'f ' + ' '.join(['%d/%d/%d' * faces.shape[1]])
            else:
                raise ValueError("invalid format for faces, allowed: (N, 3), (N, 4), (N, 9), (N,12)")
            np.savetxt(f, faces + 1, fmt=face_fmt)

