# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import csv
import numpy as np
import trimesh
import matplotlib.pyplot as plt


from models.fields import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile

from tools.logger import get_logger, get_root_logger, print_log
from tools.utils import remove_far, remove_outlier
from tools.surface_extraction import as_mesh, surface_extraction, threshold_MC
from models.VectorAdam import VectorAdam
import point_cloud_utils as pcu

import warnings
warnings.filterwarnings('ignore')


def laplacian_calculation(mesh, equal_weight=True):
    """
    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neightbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors
    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(np.ndarray)

    # stack neighbors to 1D arrays
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = np.ones(3)
        # the distance from verticesex to neighbors
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        # normalize group and stack into single array
        data = np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def face_angle_weight(mesh):
    # face_weight = np.zeros(len(mesh.faces))
    # for i in range(len(face_adj)):
    #     face_weight[face_adj[i]] += facce_angles[i]

    edge_idxs = mesh.face_adjacency_edges
    invese_edge_idx = np.vstack((edge_idxs[:, 1], edge_idxs[:, 0])).T
    edge_idxs = np.concatenate((edge_idxs, invese_edge_idx), axis=0)

    facce_angles = mesh.face_adjacency_angles
    facce_angles = np.concatenate((facce_angles, facce_angles))

    ind = np.lexsort((edge_idxs[:,1], edge_idxs[:,0]))
    edge_idxs = edge_idxs[ind]
    facce_angles = facce_angles[ind]



    # facce_angles = 2 * (1 / (1 + np.exp(-facce_angles)) - 0.5)
    facce_angles = np.exp(20*facce_angles)



    count = np.zeros((len(mesh.vertices)), dtype=int)
    for idx_pair in edge_idxs:
        count[idx_pair[0]] += 1
    start = 0
    for i, c in enumerate(count):
        facce_angles[start:start+c] = facce_angles[start:start+c]/np.sum(facce_angles[start:start+c])
        start += c

    edge_idxs = torch.LongTensor(edge_idxs.T)
    facce_angles = torch.FloatTensor(facce_angles)
    s_matrix = torch.sparse.FloatTensor(edge_idxs, facce_angles, (len(mesh.vertices), len(mesh.vertices))).cuda()
    # s_matrix_sum = torch.sparse.sum(s_matrix,dim=1).to_dense()
    return s_matrix

def neighbors2tensor(mesh, max=10):
    # return N*max, by default, neighbor is max to 10
    # edge_weight will be return, too
    #
    edge_idxs = mesh.face_adjacency_edges
    invese_edge_idx = np.vstack((edge_idxs[:, 1], edge_idxs[:, 0])).T
    edge_idxs = np.concatenate((edge_idxs, invese_edge_idx), axis=0)

    facce_angles = mesh.face_adjacency_angles
    facce_angles = np.concatenate((facce_angles, facce_angles))

    ind = np.lexsort((edge_idxs[:, 1], edge_idxs[:, 0]))
    edge_idxs = edge_idxs[ind]
    facce_angles = facce_angles[ind]

    # facce_angles = np.exp(5*facce_angles) - 1
    # facce_angles = 2/(1+np.exp(-facce_angles)) -1

    # facce_angles[facce_angles <= 3] = 0
    # facce_angles[facce_angles > 3] = 1

    # count = np.zeros((len(mesh.vertices)), dtype=int)
    # for idx_pair in edge_idxs:
    #     count[idx_pair[0]] += 1
    # start = 0
    # for i, c in enumerate(count):
    #     facce_angles[start:start + c] = facce_angles[start:start + c] / np.sum(facce_angles[start:start + c])
    #     start += c

    neighbors = []
    edge_weights = []
    for i in range(len(mesh.vertices)):
        neighbors.append([])
        edge_weights.append([])

    for i in range(edge_idxs.shape[0]):
        neighbors[edge_idxs[i][0]].append(edge_idxs[i][1])
        edge_weights[edge_idxs[i][0]].append(facce_angles[i])

    max_len = 0
    for neighbor in neighbors:
        if len(neighbor) > max_len:
            max_len = len(neighbor)

    for i in range(len(neighbors)):
        pad = max_len - len(neighbors[i])
        if pad>0:
            neighbors[i].extend([-1]*pad)
            edge_weights[i].extend([0]*pad)

    return torch.LongTensor(neighbors).cuda(), torch.FloatTensor(edge_weights).cuda()


def get_face_angle_op(neighbors, edge_weights):
    # @param neighbors N*10 idx to point's k neighbors , null is represent as -1
    # @param edge_weights N*10, neighbots's weights
    point_j = []
    point_k = []
    point_i = []
    w_j = []
    w_k = []
    for j in range(neighbors.shape[1]):
        for k in range(j+1,neighbors.shape[1]):
            sub_point_j = neighbors[:, j]
            sub_point_k = neighbors[:, k]
            sub_w_j = edge_weights[:, j]
            sub_w_k = edge_weights[:, k]
            point_j.append(sub_point_j.unsqueeze(0))
            point_k.append(sub_point_k.unsqueeze(0))
            point_i.append(torch.arange(0,neighbors.shape[0],1))
            w_j.append(sub_w_j.unsqueeze(0))
            w_k.append(sub_w_k.unsqueeze(0))

    point_i = torch.concat(point_i)
    point_j = torch.concat(point_j)
    point_k = torch.concat(point_k)
    w_j = torch.concat(w_j)
    w_k = torch.concat(w_k)
    w = w_j*w_k
    # w[w<4] = 0
    # idx = torch.nonzero(w).T  # 这里需要转置一下
    # data = w[idx[0], idx[1]]
    #
    # w = torch.sparse_coo_tensor(idx, data, w.shape)
    point_j = torch.flatten(point_j)
    point_k = torch.flatten(point_k)
    point_i = torch.flatten(point_i)
    w = torch.flatten(w)
    idx = torch.where(w>4)
    point_i = point_i[idx]
    point_j = point_j[idx]
    point_k = point_k[idx]
    w = w[idx]
    return point_i, point_j, point_k, w


def get_face_angle_loss(xyz, point_i, point_j, point_k,w):
    xyz_j = xyz[point_j]
    xyz_k = xyz[point_k]
    xyz_i = xyz[point_i]
    dist = xyz_j + xyz_k - 2 * xyz_i
    face_loss = w * torch.norm(dist, dim=1)
    return face_loss.sum(dim=0)

def laplacian_step(laplacian_op,samples):
    laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3]) - samples[:, 0:3]

    return laplacian_v


def get_mid(vertices, faces):
    fvs = vertices[faces]
    re = torch.mean(fvs,dim=1)
    return re


# def calculate_cir(vertices, faces):
#     sub_a, sub_b, sub_c = get_abc(vertices,faces)
#     s = (sub_a + sub_b + sub_c)
#     return s


# def cir_loss(samples, origin_cir, faces):
#     xyz = samples
#     # xyz.requires_grad = True
#     new_cir = calculate_cir(xyz, faces)
#
#     s_loss = torch.abs(new_cir - origin_cir)
#     s_loss = s_loss.mean()
#     return s_loss


def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
    N = 64
    s = time.time()
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()

                # grad = grad_func(pts)
                # grad = F.normalize(grad, dim=2)
                # grad = grad.reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                # g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = grad
    s2 = time.time()
    print("Extracted fields in {} s".format(s2-s))
    return u,g

class Evaluator:
    def __init__(self, args, conf_path, udf_network=None, total_size=None, centers=None):
        self.device = torch.device('cuda')
        self.args = args
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dataname
        os.makedirs(self.base_exp_dir, exist_ok=True)


        # Evaluating parameters
        self.max_iter = self.conf.get_int('evaluate.max_iter')
        self.face_angle_step = self.conf.get_int("evaluate.face_angle_step")
        self.save_iter = self.conf.get_int('evaluate.save_iter')
        self.warm_up_end = self.conf.get_int('evaluate.warm_up_end')
        self.learning_rate = self.conf.get_float('evaluate.learning_rate')
        self.optimizer = None
        self.resolution = self.conf.get_int('evaluate.resolution')
        self.threshold = self.conf.get_float('evaluate.threshold')
        self.use_exist_mesh = self.conf.get_int("evaluate.use_exist_mesh")
        self.is_cut = self.conf.get_int("evaluate.is_cut")
        if self.use_exist_mesh == 0:
            self.use_exist_mesh=False
        else:
            self.use_exist_mesh=True

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.model_list = []
        self.writer = None

        # Networks
        if udf_network is None:
            self.udf_network = CAPUDFNetwork(**self.conf['model.udf_network']).to(self.device)
            checkpoint_name = self.conf.get_string('evaluate.load_ckpt')
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                    map_location=self.device)
            print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
            self.udf_network.load_state_dict(checkpoint['udf_network_fine'])
            # self.centers = checkpoint["centers"]
            # self.total_size = checkpoint["total_size"]
            self.centers = centers
            self.total_size = total_size
        else:
            print("Using exist Network")
            self.udf_network = udf_network
            self.centers = centers
            self.total_size = total_size
        # Backup codes and configs for debug
        # self.file_backup()

    def evaluate(self):
        log_file = os.path.join(os.path.join(self.base_exp_dir), 'evaluate.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        bound_min = torch.tensor([-1, -1, -1], dtype=torch.float32)
        bound_max = torch.tensor([1, 1, 1], dtype=torch.float32)
        out_dir = os.path.join(self.base_exp_dir, 'mesh')
        os.makedirs(out_dir, exist_ok=True)
        query_func = lambda pts: self.udf_network.udf(pts)
        grad_func = lambda pts: self.udf_network.gradient(pts)

        if (not self.use_exist_mesh) or ( not os.path.exists(out_dir + '/MC_mesh.obj')):
            if not os.path.exists(out_dir + '/MC_mesh.obj') and self.use_exist_mesh:
                print("No existing MC mesh, generating ...")
            # if self.conf.get_float('train.far') > 0:
            #     mesh = remove_far(point_gt.detach().cpu().numpy(), mesh, self.conf.get_float('train.far'))
            print('Extracting mesh with resolution: {}'.format(self.resolution))
            u, g = extract_fields(bound_min, bound_max, self.resolution, query_func, grad_func)
            # g[u[:,:,:]>0.02] = (0,0,0)
            # np.savez(os.path.join(self.base_exp_dir, "field.npz"), df = u, grad=g)
            # #
            # g_sclice = np.linalg.norm(g[g.shape[0]//2],axis=-1)
            # g_sclice_x = g[g.shape[0] // 2,:,:,0]
            # g_sclice_y = g[g.shape[0] // 2, :, :, 1]
            # g_sclice_z = g[g.shape[0] // 2, :, :, 2]
            # u_sclice = u[u.shape[0]//2]
            # g_sclice = g_sclice/g_sclice.max()
            # u_sclice = u_sclice/u_sclice.max()
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice.png"), g_sclice)
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice_x.png"), g_sclice_x)
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice_y.png"), g_sclice_y)
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice_z.png"), g_sclice_z)
            # plt.imsave(os.path.join(self.base_exp_dir, "ndf_sclice.png"), u_sclice)
            # print('saved sclice with resolution: {}'.format(self.resolution))
            mesh = threshold_MC(u, self.threshold, self.resolution)
        else:
            print("using exist MC mesh")
            mesh = trimesh.load_mesh(out_dir + '/MC_mesh.obj')
        xyz = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
        origin_xyz = xyz.clone()
        xyz.requires_grad = True
        origin_lap_v = None
        # origin_edg = None
        laplacian_op = laplacian_calculation(mesh).cuda()

        mesh.export(out_dir + '/MC_mesh.obj')

        self.optimizer = VectorAdam([xyz])
        # self.optimizer = torch.optim.Adam([xyz])
        num_samples = xyz.shape[0]
        max_batch = 100000
        neighbors = None
        w_l = 1000
        for it in range(self.max_iter):
            # print(it)
            # if it % (self.max_iter//4) == 0:
            #     w_l = w_l*2
            self.update_learning_rate(it)
            if it == self.face_angle_step:
                points = xyz.detach().cpu().numpy()
                mesh.vertices = points

                neighbors, edge_weights = neighbors2tensor(mesh)
                point_i, point_j, point_k, w = get_face_angle_op(neighbors, edge_weights)
                print("points_weight update")
            if it == self.max_iter-200:
                points = xyz.detach().cpu().numpy()

                normal_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces, process=False)
                normals = torch.FloatTensor(normal_mesh.vertex_normals).cuda()
                origin_points = xyz.detach().clone()
                # origin_mid = get_mid(xyz,mesh.faces).detach().clone()
            head = 0


            # if it == n - 1:
            #     use_lap_origin = False
                # print("iter: {}".format(i))

            while head< num_samples:
                sample_subset = xyz[head: min(head + max_batch, num_samples)]
                df = query_func(sample_subset)
                df_loss = df.mean()
                # print("cal complete")

                lap_v = laplacian_step(laplacian_op, xyz)
                lap_v =  lap_v
                if it % self.lap_iter == 0 and self.use_origin:
                    origin_lap_v = lap_v.detach().clone()
                # if it % self.edg_iter == 0 and self.use_edge_loss:
                #     origin_edg = calculate_cir(xyz, mesh.faces).detach().clone()
                if self.use_origin:
                    laplacian_loss = torch.norm((lap_v - origin_lap_v), dim=1) / (
                                torch.norm(origin_lap_v, dim=1) + 1e-8)
                else:
                    lap_v = torch.mul(lap_v,lap_v)
                    lap_v = lap_v[head: min(head + max_batch, num_samples)]
                    laplacian_loss = 1 * torch.sum(lap_v, dim=1)
                laplacian_loss = w_l * laplacian_loss.mean()

                # if self.use_edge_loss:
                #     edg_loss = cir_loss(xyz, origin_edg,faces=mesh.faces)
                # else:
                #     edg_loss = torch.FloatTensor(0)
                # print(laplacian_loss)
                # w = 0.5 * (math.cos((it) / (self.max_iter) * math.pi/2) + 0.2)
                loss = df_loss
                if it<=self.max_iter-200:
                    loss = loss + laplacian_loss
                    if neighbors is not None:
                        face_angle_loss = get_face_angle_loss(xyz[:,0:3], point_i, point_j, point_k, w)/xyz.shape[0]
                        loss = loss + face_angle_loss
                else:
                    pass
                    offset = xyz - origin_points
                    # offset = torch(offset,dim=-1)
                    normal_loss =  torch.norm(torch.cross(offset,normals,dim=-1),dim=-1)
                    normal_loss = normal_loss.mean()
                    loss += 1*normal_loss
                    # min_points = get_mid(xyz, mesh.faces).detach().clone()
                    # mid_head = 0
                    # mid_df_loss = None
                    # while mid_head < min_points.shape[0]:
                    #     sub_min_points = min_points[head: min(head + max_batch, min_points.shape[0])]
                    #     mid_df = query_func(sub_min_points)
                    #     if mid_df_loss is None:
                    #         mid_df_loss = mid_df.mean()
                    #     else:
                    #         mid_df_loss += mid_df.mean()
                    #     mid_head += max_batch
                    #
                    # loss += mid_df_loss
                    # new_mid = get_mid(xyz,mesh.faces)
                    # mid_loss = new_mid-origin_mid
                    # mid_loss = torch.norm(mid_loss,dim=-1)
                    # mid_loss = mid_loss.mean()
                    # loss = mid_loss + loss


                distance_loss = torch.norm(origin_xyz-xyz,dim=1)
                distance_loss[distance_loss<2.5*self.threshold] = 0
                distance_loss = distance_loss.mean()
                loss = loss + distance_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                head += max_batch
            if (it+1) % self.save_iter == 0:
                # print(" {} iteration, loss={}".format(it, loss))
                points = xyz.detach().cpu().numpy()
                mesh.vertices = points
                mesh.export(out_dir + '/{}_{}.ply'.format(self.args.dataname,it))
        final_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces, process=False)
        # mesh.apply_scale(self.total_size)
        # mesh.apply_translation(self.centers)
        if self.is_cut == 1:
            from cut_v2 import cut_mesh_v2
            final_mesh = cut_mesh_v2(final_mesh)
            if final_mesh is not None:
                final_mesh.export(out_dir + '/' + 'mesh.ply')
                return final_mesh
        final_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces, process=False)
        final_mesh.export(out_dir + '/' + 'mesh.ply')
        return final_mesh

    def update_learning_rate(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr

        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
        
if __name__ == '__main__':
    import time

    start_time = time.time()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='demo')
    args = parser.parse_args()
    args.dir_name = args.dataname
    torch.cuda.set_device(args.gpu)
    evaluator = Evaluator(args, args.conf)

    evaluator.evaluate()
    end_time = time.time()
    print("time cost: {:.2f}s".format(end_time - start_time))