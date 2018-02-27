import numpy as np


def get_mesh_edges(tris, verts):
    u = verts[tris[:,1]] - verts[tris[:,0]]
    v = verts[tris[:,2]] - verts[tris[:,0]]
    return u, v
