import numpy as np
from os import path
import re

_triangle_regex = re.compile("^f\s+([^\/\s]+)/?\S*/?\S*\s+([^\/\s]+)/?\S*/?\S*\s+([^\/\s]+)", re.MULTILINE)
_triangle_regex_all = re.compile("^f\s+([^\/\s]+)/?([^\/\s]*)/?([^\/\s]*)\s+([^\/\s]+)/?([^\/\s]*)/?([^\/\s]*)\s+([^\/\s]+)/?([^\/\s]*)/?([^\/\s]*)", re.MULTILINE)
_normal_regex = re.compile("^vn\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)
_texcoord_regex = re.compile("^vt\s+(\S+)\s+(\S+)", re.MULTILINE)
_vertex_regex = re.compile("^v\s+(\S+)\s+(\S+)\s+(\S+)", re.MULTILINE)

def load_obj(filename, load_normals=False, load_texcoords=False, load_texture=False, load_all_triangle_defs=False):
    """ load a wavefront obj file
        loads vertices into a (x,y,z) struct array and vertex indices
        into a n x 3 index array 
        only loads obj files vertex positions and also
        only works with triangle meshes """
    vertices = np.fromregex(open(filename), _vertex_regex, np.float)
    if load_normals:
        normals = np.fromregex(open(filename), _normal_regex, np.float)
    if load_texcoords:
        texcoords = np.fromregex(open(filename), _texcoord_regex, np.float)
    if load_all_triangle_defs:
        triangles = np.fromregex(open(filename), _triangle_regex_all, np.int) - 1 # 1-based indexing in obj file format!
    else:
        triangles = np.fromregex(open(filename), _triangle_regex, np.int) - 1 # 1-based indexing in obj file format!
    r = [vertices]
    if load_normals:
        r.append(normals)
    if load_texcoords:
        r.append(texcoords)
    r.append(triangles)

    if load_texture:
        from scipy.misc import imread

        tex_file = None
        for line in open(filename).xreadlines():
            if line.startswith('mtllib'):
                mtl_file = path.join(path.dirname(filename), line.strip().split()[1])
                for mtl_line in open(mtl_file).xreadlines():
                    if mtl_line.startswith('map_Kd'):
                        tex_file = mtl_line.strip().split()[1]
        if tex_file is None:
            raise IOError("Cannot read texture from %s" % filename)
        texture = imread(path.join(path.dirname(filename), tex_file))
        r.append(texture)
    return r

def save_obj(filename, vertices, triangles, normals=None, texcoords=None, texture_file=None):
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
        n = 1
        trifmt = "%d"
        if not texcoords is None:
            np.savetxt(f, texcoords, 
                       fmt="vt %f %f")
            n += 1
            trifmt += "/%d"

        if not normals is None:
            np.savetxt(f, normals, 
                       fmt="vn %f %f %f")
            n += 1
            if texcoords is None:
                trifmt += "//%d"
            else:
                trifmt += "/%d"

        if texture_file is not None:
            f.write("usemtl material_0\n\n")

        if triangles is not None and len(triangles) > 0:
            tris_duplicated = [triangles + 1] * n
            np.savetxt(f, np.dstack(tris_duplicated).reshape((-1, n*triangles.shape[-1])), 
                       fmt="f " + ' '.join([trifmt] * triangles.shape[-1]))

