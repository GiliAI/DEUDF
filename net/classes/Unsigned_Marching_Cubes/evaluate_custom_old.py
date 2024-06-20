# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
import math
from scipy.sparse import coo_matrix
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import random
from models.fields import UDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from models.VectorAdam import VectorAdam
from tools.logger import get_root_logger
from tools.surface_extraction import threshold_MC
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


def get_abc(vertices, faces):
    fvs = vertices[faces]
    eps = 0
    sub_a = fvs[:, 0, :] - fvs[:, 1, :]
    sub_b = fvs[:, 1, :] - fvs[:, 2, :]
    sub_c = fvs[:, 0, :] - fvs[:, 2, :]
    sub_a = torch.linalg.norm(sub_a, dim=1)
    sub_b = torch.linalg.norm(sub_b, dim=1)
    sub_c = torch.linalg.norm(sub_c, dim=1)
    return sub_a, sub_b, sub_c


def calculate_cir(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    s = (sub_a + sub_b + sub_c)
    return s


def calculate_s(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    p = (sub_a + sub_b + sub_c)/2

    s = p*(p-sub_a)*(p-sub_b)*(p-sub_c)
    s[s<1e-30]=1e-30

    sqrts = torch.sqrt(s)
    # print(sqrts.min())
    # print(sqrts.max())
    return sqrts

def calculate_ss(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    p = (sub_a + sub_b + sub_c)/2

    s = p*(p-sub_a)*(p-sub_b)*(p-sub_c)
    # print(sqrts.min())
    # print(sqrts.max())
    return s


def get_mid(vertices, faces):
    fvs = vertices[faces]
    re = torch.mean(fvs,dim=1)
    return re


def cir_loss(samples, origin_cir, faces):
    xyz = samples
    # xyz.requires_grad = True
    new_cir = calculate_cir(xyz, faces)

    s_loss = torch.abs(new_cir - origin_cir)
    s_loss = s_loss.mean()
    return s_loss


def extract_fields(bound_min, bound_max, resolution, query_func, is_calculate_grad=False):
    def grad_func(x):
        x.requires_grad_(True)
        y = query_func(x)
        # y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    N = 32
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
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                if is_calculate_grad:
                    grad = grad_func(pts)
                    grad = F.normalize(grad, dim=2)
                    grad = grad.reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                    g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = grad
    s2 = time.time()
    print("Extracted fields in {} s".format(s2-s))
    return u,g


class Evaluator:
    def __init__(self, dataname,query_func, conf_path="confs/base.conf", bound_min=None,bound_max=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        self.dataname = dataname
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir'] + dataname
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.out_dir = os.path.join(self.base_exp_dir, 'mesh')

        # Evaluating parameters
        self.max_iter = self.conf.get_int('evaluate.max_iter')
        self.max_batch = self.conf.get_int("evaluate.max_batch")
        self.report_freq = self.conf.get_int("evaluate.report_freq")
        self.normal_step = self.conf.get_int("evaluate.normal_step")
        if self.normal_step<=0:
            self.normal_step=None
        self.save_iter = self.conf.get_int('evaluate.save_iter')
        self.laplacian_weight = self.conf.get_int('evaluate.laplacian_weight')
        self.max_dist_threshold = self.conf.get_int('evaluate.max_dist_threshold')
        self.warm_up_end = self.conf.get_int('evaluate.warm_up_end')
        self.learning_rate = self.conf.get_float('evaluate.learning_rate')
        self.resolution = self.conf.get_int('evaluate.resolution')
        self.threshold = self.conf.get_float('evaluate.threshold')

        if bound_min is None:
            bound_min = torch.tensor([-1+self.threshold, -1+self.threshold, -1+self.threshold], dtype=torch.float32)
        if bound_max is None:
            bound_max = torch.tensor([1-self.threshold, 1-self.threshold, 1-self.threshold], dtype=torch.float32)
        if isinstance(bound_min, list):
            bound_min = torch.tensor(bound_min, dtype=torch.float32)
        if isinstance(bound_max, list):
            bound_max = torch.tensor(bound_max, dtype=torch.float32)
        if isinstance(bound_min, np.ndarray):
            bound_min = torch.from_numpy(bound_min).float()
        if isinstance(bound_max, np.ndarray):
            bound_max = torch.from_numpy(bound_max).float()
        self.bound_min = bound_min - self.threshold
        self.bound_max = bound_max + self.threshold

        self.is_cut = self.conf.get_int("evaluate.is_cut")
        self.region_rate = self.conf.get_int("evaluate.region_rate")

        self.use_exist_mesh = self.conf.get_int("evaluate.use_exist_mesh")
        if self.use_exist_mesh == 1:
            self.use_exist_mesh=True
        else:
            self.use_exist_mesh= False
        self.export_grad_field = self.conf.get_int("evaluate.export_grad_field")
        if self.export_grad_field == 1:
            self.export_grad_field=True
        else:
            self.export_grad_field= False
        if self.conf.get_int("evaluate.use_vectorAdam") == 1:
            self.use_vectorAdam = True
        else:
            self.use_vectorAdam = False
        self.optimizer = None

        # Networks
        if query_func is None:
            self.udf_network = UDFNetwork(**self.conf['model.udf_network']).to(self.device)
            checkpoint_name = self.conf.get_string('evaluate.load_ckpt')
            checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                    map_location=self.device)
            print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
            self.udf_network.load_state_dict(checkpoint['udf_network_fine'])
            self.query_func = lambda pts: self.udf_network.udf(pts)
        else:
            self.query_func = query_func


    def evaluate(self):
        # if os.path.exists(self.out_dir + '/' + '{}_{}_new.ply'.format(self.dataname,str(self.threshold))):
        #     print("Skip {}".format(self.dataname))
        #     return
        log_file = os.path.join(os.path.join(self.base_exp_dir), 'evaluate.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger


        os.makedirs(self.out_dir, exist_ok=True)
        query_func = self.query_func

        if (not self.use_exist_mesh) or ( not os.path.exists(self.out_dir + '/MC_mesh.ply')):
            if not os.path.exists(self.out_dir + '/MC_mesh.ply') and self.use_exist_mesh:
                print("No existing MC mesh, generating ...")
            print('Extracting mesh with resolution: {}'.format(self.resolution))
            u, g = extract_fields(self.bound_min, self.bound_max, self.resolution, query_func,self.export_grad_field)
            np.savez(os.path.join(self.base_exp_dir, "field_{}.npz".format(str(self.threshold))), df = u, grad=g,bound_min=self.bound_min.cpu().numpy(),bound_max=self.bound_max.cpu().numpy())
            # #
            # g_sclice = np.linalg.norm(g[g.shape[0]//2],axis=-1)
            # g_sclice_x = g[g.shape[0] // 2,:,:,0]
            # g_sclice_y = g[g.shape[0] // 2, :, :, 1]
            # g_sclice_z = g[g.shape[0] // 2, :, :, 2]
            u_sclice = u[u.shape[0]//2]
            # g_sclice = g_sclice/g_sclice.max()
            u_sclice = u_sclice/u_sclice.max()
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice.png"), g_sclice)
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice_x.png"), g_sclice_x)
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice_y.png"), g_sclice_y)
            # plt.imsave(os.path.join(self.base_exp_dir, "grad_sclice_z.png"), g_sclice_z)
            plt.imsave(os.path.join(self.base_exp_dir, "ndf_sclice.png"), u_sclice)
            # print('saved sclice with resolution: {}'.format(self.resolution))
            print("11111")
            print(self.bound_min)
            print(self.bound_max)
            mesh = threshold_MC(u, self.threshold, self.resolution, bound_min=self.bound_min, bound_max=self.bound_max)
        else:
            print("using exist MC mesh")
            mesh = trimesh.load_mesh(self.out_dir + '/MC_mesh.ply')
        mesh.export(self.out_dir + '/MC_mesh.ply')
        # self.generate_pointcloud(mesh,1000000)
        # print("finish pc generate base on double MC")
        # init points
        xyz = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
        origin_xyz = xyz.clone()
        xyz.requires_grad = True
        # set optimizer to xyz
        if self.use_vectorAdam:
            self.optimizer = VectorAdam([xyz])
        else:
            self.optimizer = torch.optim.Adam([xyz])
        # init laplacian operation
        laplacian_op = laplacian_calculation(mesh).cuda()



        num_samples = xyz.shape[0]
        neighbors = None
        mid_head = 0
        ss_value = calculate_ss(xyz, mesh.faces)
        min_face_area = torch.mean(ss_value).detach()/100
        vertex_faces = np.asarray(mesh.vertex_faces)
        face_mask = np.ones_like(vertex_faces).astype(bool)
        face_mask[vertex_faces==-1] = False
        face_count=torch.from_numpy(np.sum(face_mask,axis=1)).cuda()
        for it in range(self.max_iter):
            # print(it)
            # if it % (self.max_iter//4) == 0:
            #     w_l = w_l*2
            self.update_learning_rate(it)
            if it == self.normal_step:
                points = xyz.detach().cpu().numpy()

                normal_mesh = trimesh.Trimesh(vertices=points, faces=mesh.faces, process=False)
                normals = torch.FloatTensor(normal_mesh.face_normals).cuda()
                origin_points = get_mid(xyz,mesh.faces).detach().clone()
                # origin_mid = get_mid(xyz,mesh.faces).detach().clone()
            head = 0


            # if it == n - 1:
            #     use_lap_origin = False
                # print("iter: {}".format(i))
            epoch_loss = 0

            while head< num_samples:

                sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)
                df_loss = df.mean()
                loss = df_loss
                # edge_lens = torch.norm(xyz[mesh.edges][:,0]-xyz[mesh.edges][:,1],dim=1)
                # ave_edge = torch.mean(edge_lens)
                # edge_loss = torch.mean(torch.abs(edge_lens-ave_edge))
                # loss = loss + edge_loss

                # s_value = calculate_s(xyz, mesh.faces)
                #
                # s_loss = torch.relu(min_face_area - s_value)
                # s_loss  = torch.sum(s_loss)
                # #
                # loss += 1e-2 * s_loss

                # print("cal complete")



                # if self.use_edge_loss:
                #     edg_loss = cir_loss(xyz, origin_edg,faces=mesh.faces)
                # else:
                #     edg_loss = torch.FloatTensor(0)
                # print(laplacian_loss)
                # w = 0.5 * (math.cos((it) / (self.max_iter) * math.pi/2) + 0.2)

                mid_points = get_mid(xyz, mesh.faces)
                if mid_head > mid_points.shape[0]:
                    mid_head = 0
                sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
                mid_df = query_func(sub_mid_points)
                mid_df_loss = mid_df.mean()
                loss += mid_df_loss

                # s_value = calculate_s(xyz, mesh.faces)[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
                # s_loss = torch.relu(min_face_area - s_value)
                # s_loss = torch.sum(s_loss)
                # #
                # loss += 1e3 * s_loss

                if self.normal_step is not None and it <= self.normal_step:
                    s_value = calculate_s(xyz, mesh.faces)
                    # face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]
                    #
                    # face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                    # face_weight = torch.sum(face_weight,dim=1)
                    # face_weight = torch.sqrt(face_weight)
                    # face_weight = face_weight.max() / face_weight
                    face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]

                    face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                    face_weight = torch.sum(face_weight, dim=1)

                    face_weight = torch.sqrt(face_weight.detach())
                    face_weight = face_weight.max() / face_weight

                    lap_v = laplacian_step(laplacian_op, xyz)
                    lap_v = torch.mul(lap_v, lap_v)
                    lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
                    laplacian_loss =  face_weight * torch.sum(lap_v, dim=1)
                    laplacian_loss = self.laplacian_weight * laplacian_loss.mean()
                    loss = loss + laplacian_loss

                self.optimizer.zero_grad()
                epoch_loss += loss.data
                loss.backward()
                self.optimizer.step()
                head += self.max_batch
                mid_head += self.max_batch
            if (it+1) % self.report_freq == 0:
                print(" {} iteration, loss={}".format(it, epoch_loss))
            if (it+1) % self.save_iter == 0:
                # print(" {} iteration, loss={}".format(it, loss))
                points = xyz.detach().cpu().numpy()
                mesh.vertices = points
                mesh.export(self.out_dir + '/{}_{}_Optimize_{}.ply'.format(self.dataname,it,str(self.threshold)))
        final_mesh = trimesh.Trimesh(vertices=xyz.detach().cpu().numpy(), faces=mesh.faces, process=False)
        # mesh.apply_scale(self.total_size)
        # mesh.apply_translation(self.centers)
        if self.is_cut == 1:
            from cut_v2 import cut_mesh_v2
            final_mesh_cuple = cut_mesh_v2(final_mesh,self.region_rate)
            if final_mesh_cuple is not None:
                final_mesh_1 = final_mesh_cuple[0]
                final_mesh_2 = final_mesh_cuple[1]
                final_mesh_1.export(self.out_dir + '/' + '{}-0_{}_new.ply'.format(self.dataname,str(self.threshold)))
                final_mesh_2.export(self.out_dir + '/' + '{}-1_{}_new.ply'.format(self.dataname,str(self.threshold)))
                loss1 = self.compute_df(final_mesh_1)
                loss2 = self.compute_df(final_mesh_2)
                if loss1<loss2:
                    final_mesh = final_mesh_1
                else:
                    final_mesh = final_mesh_2
                print("exported result")
                final_mesh.export(self.out_dir + '/' + '{}_{}_new.ply'.format(self.dataname,str(self.threshold)))
            else:
                print("It seems that model is too complex, cutting failed. Or just rerunning to try again.")
        else:
            final_mesh.export(self.out_dir + '/' + '{}_{}_new.ply'.format(self.dataname,str(self.threshold)))

        return final_mesh

    def update_learning_rate(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr

        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def generate_pointcloud(self,mesh, sample_num=20000000):
        num = 0
        samples_cpu = np.zeros((0, 3))
        init_v = torch.from_numpy(np.asarray(mesh.vertices)).cuda()
        while samples_cpu.shape[0]<sample_num:
            if init_v.shape[0]>self.max_batch:
                indice = random.sample(range(init_v.shape[0]), self.max_batch)
                indice = torch.tensor(indice)
                points = init_v[indice]
            else:
                points = init_v
            noise = 0.005*torch.randn_like(points)
            samples = (points + noise).float()
            samples.requires_grad = True
            for i in range(10):
                df_pred = self.query_func(samples)
                df_pred.sum().backward()

                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                samples = samples - F.normalize(gradient, dim=1) * df_pred.flatten()[:,None]  # better use Tensor.copy method?
                samples = samples.detach()
                samples.requires_grad = True
            df_pred = self.query_func(samples)
            samples_cpu = np.vstack((samples_cpu, samples[(df_pred.flatten() < 0.005).squeeze()].detach().cpu().numpy()))
        pc = trimesh.PointCloud(samples_cpu)
        pc.export(self.out_dir + '/' + '{}_pc.ply'.format(self.dataname))

    def compute_df(self,mesh):
        xyz = torch.from_numpy(mesh.sample(100000).astype(np.float32)).cuda()
        xyz.requires_grad = True
        num_samples = xyz.shape[0]
        head = 0
        loss = 0
        while head < num_samples:
            sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
            df = self.query_func(sample_subset)
            df_loss = df.sum().data
            # print("cal complete")

            # if self.use_edge_loss:
            #     edg_loss = cir_loss(xyz, origin_edg,faces=mesh.faces)
            # else:
            #     edg_loss = torch.FloatTensor(0)
            # print(laplacian_loss)
            # w = 0.5 * (math.cos((it) / (self.max_iter) * math.pi/2) + 0.2)
            loss += df_loss
            head += self.max_batch
        return loss/num_samples



def main(args):
    import time

    start_time = time.time()
    args.dir_name = args.dataname
    torch.cuda.set_device(args.gpu)
    conf_path = args.conf
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    device = torch.device('cuda')
    conf = ConfigFactory.parse_string(conf_text)
    udf_network = UDFNetwork(**conf['model.udf_network']).to(device)
    checkpoint_name = conf.get_string('evaluate.load_ckpt')
    base_exp_dir = conf['general.base_exp_dir'] + args.dataname
    checkpoint = torch.load(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name),
                            map_location=device)
    print(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name))
    udf_network.load_state_dict(checkpoint['udf_network_fine'])

    mesh_path = os.path.join(conf['dataset'].data_dir, "input", "{}.ply".format(args.dataname))
    mesh = trimesh.load_mesh(mesh_path)
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)
    object_bbox_min = np.array(mesh.bounds[0]) - 0.05
    object_bbox_max = np.array(mesh.bounds[1]) + 0.05

    evaluator = Evaluator(args.dataname,lambda pts: udf_network.udf(pts),  conf_path=args.conf,
                          bound_min=object_bbox_min,bound_max=object_bbox_max)

    with torch.autograd.detect_anomaly():
        evaluator.evaluate()
    end_time = time.time()
    print("time cost: {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='demo')
    args = parser.parse_args()
    main(args)