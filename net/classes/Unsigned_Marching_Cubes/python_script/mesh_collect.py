import trimesh
import os
import shutil


root = "../experiment/out_cloth_batch"
out_path = "../experiment/select_mesh"
os.makedirs(out_path, exist_ok=True)
# os.makedirs(os.path.join(out_path, "../mesh"), exist_ok=True)
# os.makedirs(os.path.join(out_path, "npz_file"), exist_ok=True)
for path in os.listdir(root):
    if path == "checkpoints":
        continue
    # mesh = trimesh.load_mesh()
    # mesh.export()
    if os.path.exists(os.path.join(root, path, "mesh/{}_399_Optimize.ply".format(path))):
        shutil.copy(os.path.join(root, path, "mesh/{}_399_Optimize.ply".format(path)), os.path.join(out_path, path + ".ply"))
    else:
        print("no mesh at {}".format(path))
    # shutil.copy(os.path.join(root, path,"mesh", "256.npz"), os.path.join(out_path, "npz_file",  path+".npz"))
