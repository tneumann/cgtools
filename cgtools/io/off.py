import numpy as np


def save_off(filename, vertices=None, faces=None, scalars=None, vmin=None, vmax=None, colors=None):
    if vertices is None:
        vertices = []
    if faces is None:
        faces = []
    has_color = scalars is not None or colors is not None
    with open(filename, 'w') as f:
        f.write("%s\n%d %d 0\n" % (['OFF', 'COFF'][has_color], len(vertices), len(faces)))
        if len(vertices) > 1:
            if has_color:
                if scalars is not None:
                    import matplotlib as mpl
                    import matplotlib.cm as cm

                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    rgba = cm.ScalarMappable(norm=norm, cmap=cm.jet).to_rgba(scalars)

                elif colors is not None:
                    rgba = colors
                    if rgba.dtype != np.uint8:
                        rgba = (rgba * 255).astype(np.uint8)

                np.savetxt(f, np.hstack((vertices, rgba[:, :3])), fmt="%.12f %.12f %.12f %d %d %d")
            else:
                np.savetxt(f, vertices, fmt="%.12f %.12f %.12f")
        if len(faces) > 1:
            for face in faces:
                fmt = " ".join(["%d"] * (len(face) + 1)) + "\n"
                f.write(fmt % ((len(face),) + tuple(map(int, face))))

def load_off(filename, no_colors=False):
    lines = open(filename).readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    assert lines[0].strip() in ['OFF', 'COFF'], 'OFF header missing'
    has_colors = lines[0].strip() == 'COFF'
    n_verts, n_faces, _ = list(map(int, lines[1].split()))
    vertex_data = np.fromstring(
        ''.join(lines[2:2 + n_verts]), 
        sep=' ', dtype=np.float).reshape(n_verts, -1)
    if n_faces > 0:
        faces = np.fromstring(''.join(lines[2+n_verts: 2+n_verts+n_faces]), 
                              sep=' ', dtype=np.int).reshape(n_faces, -1)[:, 1:]
    else:
        faces = None
    if has_colors:
        colors = vertex_data[:,3:].astype(np.uint8)
        vertex_data = vertex_data[:,:3]
    else:
        colors = None
    if no_colors:
        return vertex_data, faces
    else:
        return vertex_data, colors, faces
