import trimesh
from mesh_to_sdf import mesh_to_sdf
import torch
import numpy as np
import igl

# mesh = trimesh.load("postprocess/asian_dragon.off")
# total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
# centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
#
# mesh.apply_translation(-centers)
# mesh.apply_scale(1 / total_size)
# resolution = 256
# bound_min = [-1,-1,-1]
# bound_max = [1,1,1]
# X = torch.linspace(bound_min[0], bound_max[0], resolution)
# Y = torch.linspace(bound_min[1], bound_max[1], resolution)
# Z = torch.linspace(bound_min[2], bound_max[2], resolution)
# xx, yy, zz = torch.meshgrid(X, Y, Z)
# pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
# pts = pts.numpy()
#
# df = np.abs(igl.signed_distance(pts, mesh.vertices, mesh.faces)[0])
# df = df.reshape(resolution,resolution,resolution)
# np.savez("gt.npz", df = df)
# print(pts.shape)

# mesh = trimesh.load_mesh("D:\\Projects\\CXH\\NDC\\examples\\asian_dragon_normalized.ply")
# v = mesh.vertices
# noise = np.random.normal(0.0,0.5, v.shape[0]*3).reshape(v.shape)
# v = v+noise
# mesh.vertices = v
# mesh.export("asian_dragon_noise.ply")
pc = trimesh.load("dc.ply")
coords = torch.from_numpy(np.array(pc.vertices))
coord_max = torch.max(coords, axis=0, keepdims=True)[0]
coord_min = torch.min(coords, axis=0, keepdims=True)[0]
coord_center = 0.5*(coord_max + coord_min)
coords -= coord_center
scale = torch.norm(coords,dim=1).max()
coords /= scale
bbox_size = 2
padding = 0.0
coords *= (bbox_size/2 * (1 - padding))
mesh = trimesh.Trimesh(coords.numpy(),pc.faces)
mesh.export("gt_dc_mesh.ply")
pc = mesh.sample(8000000)
pc = trimesh.PointCloud(pc)
pc.export("dc_normalized.ply")
print("111")
resolution = 256
bound_min = [-1,-1,-1]
bound_max = [1,1,1]
X = torch.linspace(bound_min[0], bound_max[0], resolution)
Y = torch.linspace(bound_min[1], bound_max[1], resolution)
Z = torch.linspace(bound_min[2], bound_max[2], resolution)
xx, yy, zz = torch.meshgrid(X, Y, Z)
pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
pts = pts.numpy()

df = np.abs(igl.signed_distance(pts, mesh.vertices, mesh.faces)[0])
df = df.reshape(resolution,resolution,resolution)
np.savez("gt_dc.npz", df = df)
print(pts.shape)