import numpy as np
from trimesh import creation, transformations
import trimesh
import os
out = "postprocess/sep_data"
root = "data/big_scence/input"


mesh_name = "postprocess/wait_for_sep/bedroomgt_sample.ply"
# mesh_name = "result.ply"
mesh = trimesh.load_mesh(mesh_name)
rot_matrix = transformations.rotation_matrix(np.pi/3, [0,0,1], [0,0,0])
mesh.apply_transform(rot_matrix)
pointcloud = np.asarray(mesh.vertices)
shape_scale = np.max(
    [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
     np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
pointcloud = pointcloud - shape_center
pointcloud = pointcloud / shape_scale
mesh = trimesh.PointCloud(pointcloud)
rot_matrix = transformations.rotation_matrix(np.pi/3, [0,0,1], [0,0,0])
inv_rot_matrix = transformations.rotation_matrix(-np.pi/3, [0,0,1], [0,0,0])
for path in os.listdir(root):
    sub_mesh = trimesh.load_mesh(os.path.join(root,path))

    v = np.asarray(sub_mesh.vertices)
    v = v[v[:,1]>mesh.bounds[0][1]+0.15]
    v = v[v[:,2]<mesh.bounds[1][2]-0.1]
    sub_mesh = trimesh.PointCloud(v,process=False)
    # sub_mesh.apply_transform(inv_rot_matrix)
    sub_mesh.export(os.path.join(out,path))
# mesh_name = "result.ply"
# ours_mesh = trimesh.load_mesh(mesh_name)
# rot_matrix = transformations.rotation_matrix(np.pi/3, [0,0,1], [0,0,0])
# ours_mesh.apply_transform(rot_matrix)
# ours_mesh = trimesh.intersections.slice_mesh_plane(ours_mesh, (0, 0, -1), (0, 0, mesh.bounds[1][2]-0.1))
# ours_mesh = trimesh.intersections.slice_mesh_plane(ours_mesh, (0, 1, 0), (0, mesh.bounds[0][1]+0.15, 0))
# ours_mesh.export("ours.ply")
