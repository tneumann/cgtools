import numpy as np
from cgtools.vis.correspondences import visualize_mesh_morph


if __name__ == "__main__":
    x, y = list(map(np.ravel, np.mgrid[:10:20j, :10:20j]))
    z1 = np.sin(0.5 * x) + np.cos(1.2 * y)
    z2 = -0.5 * np.sin(0.4 * x) + 0.5 * np.cos(1.0 * y)
    ix = np.arange(len(x)).reshape(20, 20)
    quads = np.column_stack(list(map(np.ravel, [ix[:-1, :-1], 
                                           ix[ 1:, :-1], 
                                           ix[ 1:, 1:], 
                                           ix[:-1, 1:]])))
    tris = quads[:, [0, 1, 2,  2, 3, 0]].reshape(-1, 3)

    visualize_mesh_morph(
        np.column_stack((x, y, z1)), 
        np.column_stack((x, y, z2)), 
        quads,
        actor_property={'edge_visibility': True},
    )
    
