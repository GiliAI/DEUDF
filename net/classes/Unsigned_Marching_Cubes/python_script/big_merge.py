import trimesh
import os
import shutil


root = "postprocess/sep_data"
all_mesh = None
for path in os.listdir(root):
    print(path)
    if path == "checkpoints":
        continue
    mesh = trimesh.load_mesh(os.path.join(root,path))
    if isinstance(mesh, trimesh.Scene):
        continue
    if all_mesh is None:
        all_mesh=mesh
    else:
        all_mesh = all_mesh + mesh
all_mesh.export("gt.ply")
