import trimesh
import numpy as np

pc = trimesh.load("../data/cloth/input/165.ply")
noise = 0.005*np.random.randn(pc.vertices.shape[0],3)
np.asarray(pc.vertices)[:] += noise
pc.export("../data/cloth/input/165_noise.ply")