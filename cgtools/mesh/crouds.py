import numpy as np


def merge_meshes(list_of_verts, list_of_faces):
    merged_verts = np.vstack(list_of_verts)
    if isinstance(list_of_faces, np.ndarray) and list_of_faces.ndim == 2:
        list_of_faces = [list_of_faces] * len(list_of_verts)
    merged_tris = []
    n = 0
    for verts, tris in zip(list_of_verts, list_of_faces):
        merged_tris.append(tris + n)
        n += len(verts)
    return merged_verts, np.vstack((merged_tris))


def distribute_points(list_of_points, axes=(0, 2), n1=None, pad_factor=1.2, spacing=None):
    if isinstance(axes, int):
        axes = [axes]
    if len(axes) > 2:
        raise ValueError("axes should be a tuple with at most 2 elements")
    if spacing is None:
        spacing = np.max([np.ptp(p, axis=0) for p in list_of_points], axis=0) * pad_factor

    rng = [[0], [0], [0]]
    if len(axes) == 1:
        n1 = len(list_of_points)
    else:
        if n1 is None:
            n1 = int(np.ceil(np.sqrt(len(list_of_points))))
            n2 = int(np.ceil(np.sqrt(len(list_of_points))))
        else:
            n2 = int(np.ceil(len(list_of_points) / float(n1)))
            assert n1 * n2 >= len(list_of_points)

        rng[axes[1]] = np.r_[ : n2 * spacing[axes[1]] : spacing[axes[1]]]

    rng[axes[0]] = np.r_[ : n1 * spacing[axes[0]] : spacing[axes[0]]]

    offsets = np.column_stack((list(map(np.ravel, np.meshgrid(*rng)))))
    offsets = offsets[:len(list_of_points)]

    ret = [p + o for p, o in zip(list_of_points, offsets)]
    return ret, spacing, offsets


def distribute_meshes(list_of_points, list_of_faces, **kwargs):
    list_points_distr, spacing, offsets = distribute_points(list_of_points, **kwargs)
    return list(merge_meshes(list_points_distr, list_of_faces)) + [spacing, offsets]


def duplicate_and_distribute_mesh(verts, faces, n, **kwargs):
    return distribute_meshes([verts] * n, faces, **kwargs)


class Croud():
    def __init__(self, list_of_points, **kwargs_for_distribute):
        _, _, self.offsets = distribute_points(list_of_points, **kwargs_for_distribute)

    def distribute(self, list_of_points, list_of_faces=None):
        if len(list_of_points) != len(self.offsets):
            raise ValueError("cannot distribute a %d meshes in a croud that was set up with %d meshes" %
                             (len(list_of_points), len(self.offsets)))
        distributed_verts = [p + o for p, o in zip(list_of_points, self.offsets)]
        if list_of_faces is None:
            return np.vstack(distributed_verts)
        else:
            return merge_meshes(distributed_verts, list_of_faces)
