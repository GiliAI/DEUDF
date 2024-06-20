import torch
import torch.nn.functional as F
import numpy as np
import os
# import igl
import trimesh

from scipy.spatial import cKDTree
import sys
# sys.path.append(f"D:\\Projects\\CXH\\DeepSDF\\py_pcl\\out\\build\\x64-Debug")

# import example


def export_point_cloud(points, ndf, name):
    # data = data[tmp_index]
    color = np.absolute(ndf)
    eps = 0.00000001
    cmax = np.max(color)
    cmin = np.min(color) - eps
    clen = cmax - cmin

    color = color - cmin
    normalizzato = np.absolute(color / clen)
    # data = data[data[:, 0] > 0]

    rgb = np.ones((normalizzato.shape[0], 3))
    for i, l in enumerate(normalizzato):
        if l >0.1:
            rgb[i] = (1,0,0)
        else:
            rgb[i] = (0,1,0)
    rgb = rgb * 255
    pc = trimesh.points.PointCloud(points, rgb)
    pc.export(name)


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
        gt_sample = trimesh.points.PointCloud(gt_points)
        gt_sample.export(os.path.join(data_dir, 'data_visualize', dataname + "gt_sample.ply"))
    else:
        mesh = as_mesh(mesh)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)

        gt_points = mesh.sample(2000000)
        # deformed = example.MLS(gt_points, 3, 1, True, True)
        gt_sample = trimesh.points.PointCloud(gt_points)
        #
        gt_sample.export(os.path.join(data_dir, 'data_visualize', dataname + "_gt_sample.ply"))


    sample_num = sample
    sample_std_dev = [0.1,0.003,0.02,0.0005]
    sample_rate = np.array([0.2,0.8,0.6,0.9])
    sample_points = sample_num * sample_rate
    all_points = []
    all_df = []
    all_grad = []
    ptree = cKDTree(gt_points)
    for sigma, points_num in zip(sample_std_dev,sample_points):
        print("sampling.....")
        idx = np.random.choice(gt_points.shape[0], int(points_num), replace=True)

        points = gt_points[idx]
        boundary_points = points + sigma * np.random.randn(int(points_num), 3)
        # all_points.append(points.copy())
        # all_df.append(np.zeros_like(df))
        all_points.append(boundary_points.copy())
        df, near_idx = ptree.query(boundary_points)
        # near_grad = gt_points[near_idx] - boundary_points
        # near_grad = near_grad/np.repeat(np.linalg.norm(near_grad,axis=1),3).reshape(-1,3)
        # mls_idx = np.where(df<0.01)
        # print("{} points need to calculate MLS".format(len(mls_idx[0])))
        # df = np.abs(igl.signed_distance(boundary_points, mesh.vertices, mesh.faces)[0])
        # df = np.abs(mesh_to_sdf(mesh,boundary_points))
        # boundary_points = boundary_points[mls_idx]
        # deformed = example.MLS(gt_points,3,0.02,True,False)
        #
        # mls_dist = np.linalg.norm(deformed[:,:3]-boundary_points,axis=1)
        # df[mls_idx] = mls_dist
        # # deformed[:,3] = np.sqrt(deformed[:,3])


        # print("{} points replace MLS dist with Nearest Dist".format(np.sum(df>deformed[:,3])))
        # df[df>deformed[:,3]] = deformed[df>deformed[:,3],3]
        # df[deformed[:,3]>truncated_dist] = deformed[deformed[:,3]>truncated_dist,3]
        # all_grad.append(near_grad)

        all_df.append(df)
    all_points = np.concatenate(all_points, axis=0)
    all_df = np.concatenate(all_df, axis=0)
    # all_grad = np.concatenate(all_grad, axis=0)
    all_df[all_df>truncated_dist] = truncated_dist

    os.makedirs(os.path.join(data_dir, 'query_data'), exist_ok=True)
    np.savez(os.path.join(data_dir, 'query_data', dataname)+'.npz', sample = all_points, ndf = all_df)
    export_point_cloud(all_points, all_df, os.path.join(data_dir, 'data_visualize', dataname)+'.ply')
    return mesh, centers, total_size


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
        self.mesh, self.centers, self.total_size = process_data(self.data_dir, dataname)

        load_data = np.load(os.path.join(self.data_dir, 'query_data', self.data_name))
        
        self.samples = np.asarray(load_data['sample']).reshape(-1, 3)
        self.ndf = np.asarray(load_data['ndf']).reshape(-1,1)
        # self.grad = np.asarray(load_data['grad']).reshape(-1, 3)
        self.sample_points_num = self.samples.shape[0]

        self.samples = torch.from_numpy(self.samples).to(self.device).float()
        self.ndf = torch.from_numpy(self.ndf).to(self.device).float()
        # self.grad = torch.from_numpy(self.grad).to(self.device).float()
        self.iter = 0
        
        print(f'NP data End with {self.sample_points_num} points')

    def get_train_data(self, batch_size):
        index_fine = np.random.choice(self.sample_points_num, batch_size, replace = True)
        index = index_fine
        samples = self.samples[index]
        ndf = self.ndf[index]
        # grad = self.grad[index]
        return samples, ndf

    def gen_new_data_from_mesh(self, mesh,point_num=None):
        if point_num is None:
            point_num = self.sample_points_num
        print("Generating New Data from Mesh...")
        points = mesh.sample(point_num)
        df = np.exp(2*np.abs(igl.signed_distance(points, self.mesh.vertices, self.mesh.faces)[0]))-1
        new_samples = np.asarray(points).reshape(-1, 3)
        new_ndf = np.asarray(df).reshape(-1)
        new_samples = new_samples[new_ndf>0.001]
        new_ndf = new_ndf[new_ndf>0.001]
        new_ndf = new_ndf.reshape(-1,1)
        print("Generated {} points".format(new_ndf.shape[0]))
        new_samples = torch.from_numpy(new_samples).to(self.device).float()
        new_ndf = torch.from_numpy(new_ndf).to(self.device).float()
        export_point_cloud(new_samples.cpu().numpy(), new_ndf.cpu().numpy(),
                           os.path.join(self.data_dir, 'query_data', self.data_name) + '_enhanced.ply')
        self.samples = torch.cat((self.samples, new_samples),0)
        self.ndf = torch.cat((self.ndf,new_ndf),0)
        self.sample_points_num = self.ndf.shape[0]




    
