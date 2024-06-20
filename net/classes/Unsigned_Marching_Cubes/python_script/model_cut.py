#删除一部分mesh用于展示内部结构
import matplotlib.pyplot as plt

import trimesh
import numpy as np
import os


in_root = "../postprocess/wait_for_cut_"
out_root = "../postprocess/cutted_mesh"
os.makedirs(out_root, exist_ok=True)
for mesh_name in os.listdir(in_root):
    count = 0
    mesh = trimesh.load_mesh(os.path.join(in_root, mesh_name))
    # total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    # centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    # mesh.apply_translation(-centers)
    # mesh.apply_scale(1 / total_size)
    # pc = np.asarray(mesh.vertices)
    # pc = pc[pc[:,0]<0]
    mesh_1 = trimesh.intersections.slice_mesh_plane(mesh, (0,0,1), (0,0,0.05))
    mesh_2 = trimesh.intersections.slice_mesh_plane(mesh, (-1, 0, 0), (-0.05, 0, 0))
    mesh_2 = trimesh.intersections.slice_mesh_plane(mesh_2, (0,0,-1), (0,0,0.05))
    # mesh_3 = trimesh.intersections.slice_mesh_plane(mesh, (0, 0, -1), (0, 0, 0.0))
    # mesh.vertices[:,2] = -mesh.vertices[:,2]
    mesh= mesh_1 + mesh_2
    # mesh.merge_vertices()
    # mesh.remove_duplicate_faces()
    mesh.export(os.path.join(out_root, mesh_name))
