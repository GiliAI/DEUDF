import trimesh
from trimesh import creation, transformations
import numpy as np
import open3d as o3d

# mesh_name = "D:\\render_data\\all_data_for_render\\compare_mesh\\car1_ours_ndf.ply"
# mesh = trimesh.load_mesh(mesh_name)
#
# rot_matrix = transformations.rotation_matrix(np.pi/2, [0,1,0], [0,0,0])
# mesh.apply_transform(rot_matrix)
# mesh.apply_scale(0.625)
# mesh.vertices[:,2] = -mesh.vertices[:,2]
# mesh.export("car1_ours_ndf.ply")
mesh_name = "D:\\render_data\\all_data_for_render\\high-res-sdf\\dragon-idf-mc1024.ply"
mesh = o3d.io.read_triangle_mesh(mesh_name)
vertices = np.asarray(mesh.vertices)
vertices += 1 / 1024
mesh.scale(1024/1023, center=(0,0,0))
o3d.io.write_triangle_mesh("1024.ply",mesh)
# mesh.export("1024.ply")
# mesh = trimesh.load_mesh("postprocess/cutted_mesh/dragon-gt.ply")
# pc = trimesh.PointCloud(mesh.sample(3000000))
# pc.export("dragon_pc.ply")