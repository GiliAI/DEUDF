import trimesh
import numpy as np


mesh = trimesh.load_mesh("data/owndata/input/bunny_HD.ply")
pointcloud = mesh.sample(1000000)
shape_scale = np.max(
    [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
     np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
pointcloud = pointcloud - shape_center
pointcloud = pointcloud / shape_scale
trimesh.PointCloud(pointcloud).export("bunnt_gt.ply")