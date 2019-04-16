import numpy as np


def save_pointcloud(filename, points, point_colors=None):
    """ 
    save a 3d point cloud in PLY format
    points should be given as an array (or list) of 3d positions
    point_colors can be given if desired, as an array (or list) of RGB tuples
    the PLY files can be viewed e.g. using meshlab
    """
    points = np.asanyarray(points)
    if not points.ndim == 2 and points.shape[-1] == 3:
        raise ValueError("points must be an n x 3 array of 3d point coordinates")
    if point_colors is not None:
        point_colors = np.asanyarray(point_colors)
        if not point_colors.shape == points.shape:
            raise ValueError("point_colors must be an n x 3 array of rgb values with the same size as points")
    header = [
        "ply",
        "format ascii 1.0",
        "element face 0",
        "property list uchar int vertex_indices",
        "element vertex %d" % len(points),
        "property float x",
        "property float y",
        "property float z",
    ]
    if point_colors is not None:
        header += [
            "property uchar diffuse_red",
            "property uchar diffuse_green",
            "property uchar diffuse_blue",
        ]
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\nend_header\n')
        if points is not None and len(points) > 0:
            data_channels = np.hsplit(points, 3)
            fmt = "%g %g %g"
            if point_colors is not None:
                data_channels += np.hsplit(point_colors, 3)
                fmt += " %d %d %d"
            data = np.hstack(data_channels)
            np.savetxt(f, data, fmt=fmt)

def load_pointcloud(filename):
    with open(filename) as f:
        line = None
        has_color = False
        while line != "end_header":
            line = f.readline().strip()
            if line.startswith("property uchar diff"):
                has_color = True
        data = np.loadtxt(f)
        if has_color:
            return data[:,:3], data[:,3:]
        else:
            return data


