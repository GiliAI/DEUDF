import torch
import torch.nn.functional as F
import numpy as np
import os
# import igl
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import sys
# sys.path.append(f"D:\\Projects\\CXH\\DeepSDF\\py_pcl\\out\\build\\x64-Debug")

# import example


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


def process_data(data_dir, dataname, sample=100000):
    truncated_dist = 0.05
    if os.path.exists(os.path.join(data_dir, 'input', dataname+ ".ply")):
        os.makedirs(os.path.join(data_dir, 'data_visualize'), exist_ok=True)
        mesh = trimesh.load(os.path.join(data_dir, 'input', dataname) + ".ply")
    elif os.path.exists(os.path.join(data_dir, 'input', dataname+ ".obj")):
        os.makedirs(os.path.join(data_dir, 'data_visualize'), exist_ok=True)
        mesh = trimesh.load(os.path.join(data_dir, 'input', dataname) + ".obj")
    else:
        print('Only support .ply or .obj')
        exit()

    if isinstance(mesh, trimesh.PointCloud):
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        gt_points = np.asarray(mesh.vertices)
    else:
        mesh = as_mesh(mesh)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        grid_points = create_grid_points_from_bounds(-1, 1, 256)
        kdtree = cKDTree(grid_points)
        gt_points = mesh.sample(100000)

    occupancies = np.zeros(len(grid_points), dtype=np.int8)

    _, idx = kdtree.query(gt_points)
    occupancies[idx] = 1
    return np.reshape(occupancies, (256,) * 3)


class Dataset:
    def __init__(self, conf, dataname):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.data_name = dataname + '.npz'

        # if os.path.exists(os.path.join(self.data_dir, 'query_data', self.data_name)):
        #     print('Query data existing. Loading data...')
        # else:
        print('Processing data...')
        self.occ = process_data(self.data_dir, dataname)

        self.occ = torch.from_numpy(self.occ).to(self.device).float()
        # self.grad = torch.from_numpy(self.grad).to(self.device).float()
        self.iter = 0
        self.object_bbox_min = [-1.0,-1.0,-1.0]
        self.object_bbox_max = [1.0, 1.0, 1.0]
        #
        # print(f'NP data End with {self.sample_points_num} points')

    def get_test_data(self):
        return self.occ




    
