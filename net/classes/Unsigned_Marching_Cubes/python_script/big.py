import numpy as np
from trimesh import creation, transformations
import trimesh


# mesh_name = "postprocess/wait_for_sep/bedroomgt_sample.ply"
mesh_name = "../Armadillo_399_Optimize.ply"
mesh = trimesh.load_mesh(mesh_name)

rot_matrix = transformations.rotation_matrix(-np.pi/2, [0,1,0], [0,0,0])
mesh.apply_transform(rot_matrix)
# rot_matrix = transformations.rotation_matrix(np.pi/2, [1,0,0], [0,0,0])
# mesh.apply_transform(rot_matrix)
# v = np.asarray(mesh.vertices)
# v = v[v[:,1]>mesh.bounds[0][1]+0.1]
# v = v[v[:,2]<mesh.bounds[1][2]-0.1]
# mesh = trimesh.PointCloud(v)
mesh.export("../Armadillo_399_Optimize-n.ply")
# mesh = trimesh.intersections.slice_mesh_plane(mesh, (0, 0, -1), (0, 0, mesh.bounds[1][2]-0.1))
# mesh = trimesh.intersections.slice_mesh_plane(mesh, (0, 1, 0), (0, mesh.bounds[0][1]+0.1, 0))
# mesh.export("ours.ply")
