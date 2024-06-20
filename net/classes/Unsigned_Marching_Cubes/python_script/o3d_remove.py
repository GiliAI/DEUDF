import os

import numpy as np
import open3d as o3d
import copy


exp_root = "../experiment/out_cloth_batch"
for name in os.listdir(exp_root):
    mesh_name = os.path.join(exp_root,name,"mesh", "{}_399_Optimize.ply".format(name))
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)

    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < cluster_n_triangles[triangle_clusters].max()
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    o3d.io.write_triangle_mesh(mesh_name[:-4]+"_clean.ply",mesh_0)