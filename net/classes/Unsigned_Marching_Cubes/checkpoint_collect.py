import trimesh
import os
import shutil


root = "experiment/outs_batch"
out_path = "select_data"
data_name = "ckpt_080000.pth"
os.makedirs(out_path, exist_ok=True)

for path in os.listdir(root):
    os.makedirs(os.path.join(out_path,path, "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(root, path, "checkpoints",data_name), os.path.join(out_path, path, "checkpoints",data_name))
