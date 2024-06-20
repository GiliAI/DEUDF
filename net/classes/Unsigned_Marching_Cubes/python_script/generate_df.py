import os
import torch
import trimesh
import numpy as np
from mesh_to_sdf import mesh_to_sdf
from skimage import measure


N = 32
resolution = 96
root = "postprocess/wait_for_df"
out_root = "postprocess/df_data"
os.makedirs(out_root,exist_ok=True)
for p in os.listdir(root):
    mesh = trimesh.load_mesh(os.path.join(root,p))
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)
    X = torch.linspace(-1, 1, resolution).split(N)
    Y = torch.linspace(-1, 1, resolution).split(N)
    Z = torch.linspace(-1, 1, resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                print("11111")
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).numpy()
                val = mesh_to_sdf(mesh, pts).reshape((32,32,32))
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    u = np.abs(u)
    vertices, triangles, _, _ = measure.marching_cubes(
        u, 0.05, spacing=(2 / resolution, 2 / resolution, 2 / resolution))
    vertices -= 1
    # t = vertices[:,1].copy()
    # vertices[:,1] = vertices[:,2]
    # vertices[:, 2] = -t
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.export(os.path.join(out_root,p))