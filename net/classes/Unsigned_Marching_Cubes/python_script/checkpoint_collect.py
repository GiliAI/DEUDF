import trimesh
import os
import shutil


root = "../experiment/outs_car_capudf"
out_path = "../experiment/select_data"
data_name = "ckpt_060000.pth"
os.makedirs(out_path, exist_ok=True)

for path in os.listdir(root):
    os.makedirs(os.path.join(out_path,path, "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(root, path, "checkpoints",data_name), os.path.join(out_path, path, "checkpoints",data_name))
