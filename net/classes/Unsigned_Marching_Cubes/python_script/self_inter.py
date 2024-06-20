import pymeshlab
import open3d as o3d
import trimesh
import numpy as np

mesh = trimesh.load_mesh("D:\\idf-main\\experiment\\outs_batch\\test\\mesh\\test_0.0015_new.ply")
ms = pymeshlab.MeshSet()
ms.load_new_mesh("D:\\idf-main\\experiment\\outs_batch\\test\\mesh\\test_0.0015_new.ply")
ms.compute_selection_by_self_intersections_per_face()
faces = ms.current_mesh().face_selection_array()
print(np.sum(faces))
mesh1 = trimesh.Trimesh(mesh.vertices, mesh.faces[faces])
mesh1.process()
inv_faces = np.bitwise_not(faces)
mesh2 = trimesh.Trimesh(mesh.vertices, mesh.faces[inv_faces])
mesh2.process()
mesh1.export("1-ours.ply")
mesh2.export("2-ours.ply")
