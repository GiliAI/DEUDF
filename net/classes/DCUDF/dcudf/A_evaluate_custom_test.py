# -*- coding: utf-8 -*-
import copy
import time
import torch
import torch.nn.functional as F
import math
from scipy.sparse import coo_matrix
import numpy as np
import trimesh
import matplotlib
import matplotlib.pyplot as plt
import random
from models.fields import UDFNetwork
# from models.fields_origin import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
from dcudf.VectorAdam import VectorAdam
# from tools.logger import get_root_logger
# from tools.surface_extraction import threshold_MC
from dcudf.mesh_extraction import threshold_MC
import warnings
import queue
import os
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')
    # mesh.vertices.view: 用于查看三角面网格对象中存储顶点坐标的试图（view）的操作。
    # view 返回一个numpy数组，该数组是对三角面网格对象中顶点坐标数据的视图？
    # ？

def laplacian_calculation(mesh, equal_weight=True, selected=None):
    neighbors = mesh.vertex_neighbors # 取出顶点的邻居信息，返回的类型为，(len(self.vertices), ) int
    vertices = mesh.vertices.view(np.ndarray) # 获取三角面网格的顶点坐标，并将其转化为numpy
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])
    # 是否采用相同的权重。
    if equal_weight:
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        ones = np.ones(3)
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        data = np.concatenate([i / i.sum() for i in norms])
    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)  # 创建一个 coo格式的稀疏矩阵，即两个点之间对应的权重值，例如：matrix[0] = [x0,y0] weight0 ...
    values = matrix.data   # 从矩阵中提取非零元素值，
    indices = np.vstack((matrix.row, matrix.col))   # 创建一个新的二维数组，其中每一列都是由 稀疏矩阵matrix中非零元素的行索引和列索引构成的。

    if selected is not None:
        # 如果selected不为None，则返回[m,n]的稀疏矩阵
        selected_indices = np.array(selected)
        mask = np.isin(indices[0], selected_indices)
        selected_indices_map = {v: i for i, v in enumerate(selected_indices)}
        selected_rows = indices[0][mask]
        selected_cols = indices[1][mask]
        selected_values = values[mask]
        
        selected_rows_mapped = [selected_indices_map[v] for v in selected_rows]
        i = torch.LongTensor([selected_rows_mapped, selected_cols])
        v = torch.FloatTensor(selected_values)
        shape = (len(selected), len(vertices))
    else:
        # 如果selected为None，则返回[n,n]的稀疏矩阵
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = matrix.shape
    test = torch.sparse.FloatTensor(i, v, torch.Size(shape))    # 创建稀疏张量,其中非零元素的值和位置与之前的稀疏矩阵matrix相同
    
    return test

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

def laplacian_step(laplacian_op,samples,selected = None):
    '''
    在对拉普拉斯的算子计算一边权值
    '''
    # torch.sparse.mm(sparse, dense): 是pytorch中的一个函数，用于计算稀疏矩阵与密集矩阵之间的乘法运算：
    #               spars: 一个稀疏矩阵，通常表示为系数张量，其中包含非零元素的索引和对应的值
    #               dense：一个密集矩阵或密集张量
    #
    # 其中： laplacian_op 是 (n,n) 的矩阵，samples 是 (n,3) 的矩阵，n个顶点， 3是xyz 
    laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3]) 
    if selected == None:
        laplacian_v = laplacian_v - samples[:, 0:3]
    else:
        laplacian_v = laplacian_v - selected[:, 0:3]

    # laplacian_v 是 拉普拉斯算子对点坐标进行加权处理后的结果，但为什么呢？
    return laplacian_v

def get_abc(vertices, faces):
    '''
        计算三角形三个边的长度
        ---
        - 0:x1,y1,z1 1:x2,y2,z2
        - 距离 d = sqrt((x1-x2)²,(y1-y2)²,(z1-z2)²)
        - sub_a = [x = x1 - x2, y = y1 - y2, z = z1 - z2]/ sub_b = .... 
        - 调用linalg.norm:sub_a = sqrt(x²+y²+z²)
    '''

    fvs = vertices[faces]                   # 获取每个面对应的顶点坐标，存储在变量fvs中。 数组模型为，[n,3,d]即，n个三角形，3个顶点，d为对应顶点的（x,y,z）坐标
    eps = 0                                 # ？

    # 分别表示面的三条边，a边，b边，c边/得到的边是一个向量。
    sub_a = fvs[:, 0, :] - fvs[:, 1, :]     # 所有三角形中，第一个顶点与第二个顶点之间的向量差。
    sub_b = fvs[:, 1, :] - fvs[:, 2, :]     # 第二个与第三个
    sub_c = fvs[:, 0, :] - fvs[:, 2, :]     # 第三个与第一个
    # torch.linalg.norm(input, ord=None,dim=None,keepdim=False, *,out=none,dtype=none)
    #       input: 输入的向量或矩阵
    #       ord: 计算范数的类型，默认none， 表示计算Frobenius范数。可以设为整数值或字符串来计算其他类型的范数。
    #       dim：指定在哪个维度上计算范数。当dim为整数时，表示计算该维度上的范数；当dim为元组或列表时，表示沿多个维度上计算范数。
    #       keepdim：bool，表示是否保持维度，如果设置true，保持范数结果的维度，维度将会是input维度+1，长度为1的维度插入到指定dim上
    #       out：输出 张量，存储结果
    #       dytype：输出结果的数据类型
    # 根据给定的参数计算输入矩阵或向量的范数
    # 计算每条边的长度/计算每个向量的长度，利用欧几里得范数
    sub_a = torch.linalg.norm(sub_a, dim=1) # 得到三角形，a边的长度
    sub_b = torch.linalg.norm(sub_b, dim=1) # 得到三角形，b边的长度
    sub_c = torch.linalg.norm(sub_c, dim=1) # 得到三角形，c边的长度
    return sub_a, sub_b, sub_c

def calculate_cir(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    s = (sub_a + sub_b + sub_c)
    return s

def calculate_s(vertice, faces):

    '''
    求三角形面积，根据求到的三角形面积
    '''
    vertices = vertice
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    p = (sub_a + sub_b + sub_c)/2

    s = p*(p-sub_a)*(p-sub_b)*(p-sub_c)
    # 将 s 中小于阈值 1e-30 的面积值设置为 1e-30 ， 避免出现负数或零面积
    s[s<1e-30]=1e-30

    # 开个根号
    sqrts = torch.sqrt(s)
    # print(sqrts.min())
    # print(sqrts.max())
    return sqrts

def calculate_angle(vertices, faces):
    """
    计算三个点之间的角度
    参数:
        a: 第一个点的坐标，形状为 (N, 3)
        b: 第二个点的坐标，形状为 (N, 3)
        c: 第三个点的坐标，形状为 (N, 3)
    返回:
        angle: 三个点之间的角度，形状为 (N,)
    """
    # 计算每个面片的三个角度
    vertex_a = vertices[faces[:, 0]]
    vertex_b = vertices[faces[:, 1]]
    vertex_c = vertices[faces[:, 2]]

    # 计算三角形的边向量
    vec_ab = vertex_b - vertex_a + 0.00001
    vec_bc = vertex_c - vertex_b + 0.00001
    vec_ca = vertex_a - vertex_c + 0.00001

    # 计算三角形的边长度平方
    length_ab_sq = torch.sum(vec_ab**2, dim=1) + 0.00001
    length_bc_sq = torch.sum(vec_bc**2, dim=1) + 0.00001
    length_ca_sq = torch.sum(vec_ca**2, dim=1) + 0.00001

    # 计算三个角的余弦值
    cos_a = torch.sum(vec_ab * vec_ca, dim=1) / (torch.sqrt(length_ab_sq * length_ca_sq) + 0.00001)
    cos_b = torch.sum(vec_bc * -vec_ab, dim=1) / (torch.sqrt(length_bc_sq * length_ab_sq) + 0.00001)
    cos_c = torch.sum(-vec_ca * -vec_bc, dim=1) / (torch.sqrt(length_ca_sq * length_bc_sq) + 0.00001)

    # 将余弦值转换为角度（弧度）
    angle_a = torch.acos(cos_a)+ 0.00001
    angle_b = torch.acos(cos_b)+ 0.00001
    angle_c = torch.acos(cos_c)+ 0.00001

    # 计算与目标角度60度的差值
    angle_diff_a = torch.abs(angle_a - 60)+ 0.00001
    angle_diff_b = torch.abs(angle_b - 60)+ 0.00001
    angle_diff_c = torch.abs(angle_c - 60)+ 0.00001

    # # 将角度转换为弧度
    # angle_a_rad = torch.mul(torch.pi / 180.0, angle_a)
    # angle_b_rad = torch.mul(torch.pi / 180.0, angle_b)
    # angle_c_rad = torch.mul(torch.pi / 180.0, angle_c)

    # # 计算与目标角度60度的差值
    # angle_diff_a = torch.abs(angle_a_rad - (torch.pi / 3))
    # angle_diff_b = torch.abs(angle_b_rad - (torch.pi / 3))
    # angle_diff_c = torch.abs(angle_c_rad - (torch.pi / 3))

    # 计算总的角度损失
    angle_loss = torch.mean(angle_diff_a + angle_diff_b + angle_diff_c + 0.00001)
    return angle_loss

def get_mid(vertices, faces):
    '''
        计算每个三角形的中点位置。
        - 返回(n,3)其中n表示n个三角形,3是坐标xyz
    '''
    fvs = vertices[faces]       # vertices:[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]],faces:[[0,1,2],[1,2,3]]
    # fvs:[
    #       [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
    #       [[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]]
    #     ]
    # (m,3,3):m是面片的数量，中间3是三个顶点，后面3是每个顶点的坐标
    re = torch.mean(fvs,dim=1)  # 对切片后的顶点位置数组沿着维度 1 ， 计算均值，得到每个面片的中点坐标。
    return re   # 返回中间点

def cir_loss(samples, origin_cir, faces):
    
    x = 0
    xyz = samples # xyz: is a 123
    # xyz.requires_grad = True
    new_cir = calculate_cir(xyz, faces)

    s_loss = torch.abs(new_cir - origin_cir)
    s_loss = s_loss.mean()
    return s_loss

def extract_fields(bound_min, bound_max, resolution, query_func, is_calculate_grad=False):

    # 计算点云在给定坐标点处的梯度，接受参数x，表示点云的坐标。
    def grad_func(x):
        
        # requires_grad_(True): 是一个PyTorch张量(Tensor)的方法，将张量标记为需要计算梯度。
        #                       其是一个就地修改的方法，将原始张量的 requires_grad 属性设置为 True
        x.requires_grad_(True)              # 自动计算梯度
        
        y = query_func(x)                   # 计算点云在给定坐标点处的值
        # y.requires_grad_(True)            # 自动计算梯度
        d_output = torch.ones_like(y, requires_grad=False, device=y.device) # 定义了一个与 y 相同的张量，作用可能是为了进行链式法则求导时的乘法运算

        # https://zhuanlan.zhihu.com/p/279758736  对torch.autograd.grad的详细讲解。
        # torch.autograd.grad: 计算梯度的函数之一，用于计算一个或多个输出，相对于输入变量的梯度。
        #                           outputs: 需要对其进行微分的 标量/张量 或张量序列
        #                           inputs : 需要对其进行微分的张量序列， 与outputs中的张量一一对应
        #                           grad_outputs: 关于outputs的梯度，形状与outputs相同。 默认为none，表示计算梯度的标量为1.
        #                           retain_graph: 是否在返回后保留计算图。 默认情况下，计算图被清除以减少内存使用
        #                           create_graph: 指定是否构建高阶导数的计算图。 默认情况下，PyTorch不会构建高阶导数的计算图。
        #                           only_inputs ：指定是否只返回输入的梯度，而不返回其他中间变量的梯度。 默认情况下， PyTorch返回与输入变量数量相同的梯度序列。
        #       
        #       x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        #       y = x**2 + 2*x + 1
        #       grad_y_x = torch.autograd.grad(y, x)
        #       print(grad_y_x)  # 输出 (tensor([4., 6., 8.]),)
        #
        #   计算 y 相对于 x 的梯度，并将结果存储在变量gradients中。
        #   其中，gradients = xxx()[0] 中的[0]表示，通过索引运算符选择梯度张量中的第一个元素
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # 通过 unsqueeze(1) 将其维度扩展为 (batch_size, 1, input_size)的形状，batch_size:输入张量的大小。input_size是输入张量的维度大小，512？
        return gradients.unsqueeze(1)

    # 对三个维度的数值范围进行均匀分割
    # torch.linspace : 用于生成等间隔的张量。
    #                       start:张量中起始值
    #                       end:张量中的结束值
    #                       steps:（可选）生成的张量中的元素个数，默认100
    #                       out:（可选）输出张量
    #                       dtype:（可选）输出张量的数据类型。如果未指定，默认为输入张量的数据类型
    #                       layout:（可选）输出张良的布局，默认为torch.strided
    #                       device:（可选）张量所在的设备，未指定，默认为当前设备
    #                       requires_grad:（可选）师傅需要计算梯度，默认胃False
    # #
    N = 32  # 指定分割块数
    s = time.time() # 获取 开始 时间戳
    # 从 -1 ~ 1 ， 坐标为 分辨率， 分割出 N 个小块
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N) # 将第一个维度，从bound_min[0] 到 bound_max[0] 进行均匀分割，生成长度为resolution的等间隔张量
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N) # 将第二个维度，从bound_min[1] 到 bound_max[1] 进行均匀分割，生成长度为resolution的等间隔张量
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N) # 将第三个维度，从bound_min[2] 到 bound_max[2] 进行均匀分割，生成长度为resolution的等间隔张量
    #split(N) 将每个维度的等间隔张量切割成N个块，返回一个，包含 N 个张量的元组， X,Y,Z，分别表示沿三个维度的均匀间隔数值。


    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    
    # enumerate(): 将一个可迭代对象（列表、元组或字符串）生成一个枚举对象，返回枚举对象包含索引和对应的元素

    # fruits = ['apple', 'banana', 'orange']
    # for index, fruit in enumerate(fruits):
    #   print(index, fruit) // 0 apple   1 banana   2 orange
    #
    # with torch.no_grad():
    # xs中包含一个小块， 一共 N 个小块，有 N 次循环，ys，zs同理
    # X 应该是一个[resolution / N, N]维的张量，而， xs应该是一个[resolution / n， 1] 的张量
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                # x 层，y 行，z 列  //meshgrid: https://blog.csdn.net/Small_Lemon_Tree/article/details/107272557
                xx, yy, zz = torch.meshgrid(xs, ys, zs) # xx, yy, zz应该是一个 8*8*8 的张量（分辨率：256，N = 32）
                # 使用reshape，将xx,yy,zz变成 （512,1）的张量
                # 再由torch.cat，拼接成一个，(512, 3) 的张量。
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                # .reshape(len(xs),len(ys),len(zs)) 得到一个维度是(8,8,8)的新张量
                # .detach() 得到一个新的没有梯度追踪的张量，将该张量与其计算图分离。
                # val 将是一个新的没有梯度的(8,8,8)numpy张量，且是从query_func中得到的距离点。
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                # 起始位置：(xi * N, yi * N, zi * N), 终止位置：如下。
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                
                # 如果需要计算梯度：
                if is_calculate_grad:
                    grad = grad_func(pts)   # 求出梯度
                    grad = F.normalize(grad, dim=2) # 使用F.normalize（）对grad进行标准化操作，其中 dim = 2 表示在第三个维度上进行标准化。
                    # val 将是一个新的没有梯度的(8,8,8)numpy张量，且是从query_func中得到的距离点。
                    grad = grad.reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                    # 起始位置：(xi * N, yi * N, zi * N), 终止位置：如下。
                    g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = grad
    s2 = time.time()    # 获取 结束 时间戳
    print("Extracted fields in {} s".format(s2-s))  # 提取用时
    # u是一个正常的结果，g是一个求导后的结果 / 精度³ 
    return u,g

class Evaluator:
    def __init__(self,query_func, resolution,threshold,#base_dir, dataname,
                 max_iter=400, normal_step=300,laplacian_weight=10, bound_min=None,bound_max=None,
                 is_cut = True, region_rate=20,
                 max_batch=500000, learning_rate=0.0005, warm_up_end=25, use_exist_mesh=True,
                 save_iter=50, report_freq=1,export_grad_field=True,
                 sub_weight = 2, local_weight = 0.1, local_it = 1001):
        self.device = torch.device('cuda')      # 设置为CUDA加速，在GPU上进行计算
        
        self.sub_weight = sub_weight # 细分程度， 值越低，细分程度越高
        print("sub:",self.sub_weight)
        self.local_it = local_it
        self.local_weight = local_weight
        # self.dataname = dataname    # 数据名称
        # self.base_dir = base_dir    # 基础目录的路径       
        # self.base_exp_dir = os.path.join(self.base_dir, dataname)   # 将基础目录和数据名称拼接成实验基础目录路径
        # os.makedirs(self.base_exp_dir, exist_ok=True)   # 存储后续生成的实验数据，该函数用于递归的创建多层目录，接受一个路径作为参数，在指定路径下创建所有缺失的目录，true：如果目标目录已经存在，则不触发FileExistsError异常
        # self.out_dir = os.path.join(self.base_exp_dir, 'mesh')  # 用于存储生成的网格数据。基础路径+mesh
        # self.out_dir = os.path.join('/pub/data/yufg/data/shapenet_car_res', dataname)
        # print(self.out_dir)

        # Evaluating parameters
        self.max_iter = max_iter                    # 最大迭代次数
        self.max_batch = max_batch                  # 最大批次数
        self.report_freq = report_freq              # 报告频率，每隔多少步报告一次loss
        self.normal_step = normal_step              # 第一轮迭代次数，
        self.save_iter = save_iter                  # 多少次迭代，对结果进行一次保存
        self.laplacian_weight = laplacian_weight    # 拉普拉斯权重
        self.warm_up_end = warm_up_end              # 让学习率逐渐增大
        self.learning_rate = learning_rate          # 学习率，模型训练中的学习率的大小
        self.resolution = resolution                # 模型分辨率：256^3\512^3
        self.threshold = threshold                  # 阈值

        if bound_min is None:
            bound_min = torch.tensor([-1+self.threshold, -1+self.threshold, -1+self.threshold], dtype=torch.float32)    # ?
        if bound_max is None:
            bound_max = torch.tensor([1-self.threshold, 1-self.threshold, 1-self.threshold], dtype=torch.float32)       # ?
        if isinstance(bound_min, list):
            bound_min = torch.tensor(bound_min, dtype=torch.float32)    # 如果bound_min类型是list，将其转换为PyTorch张量
        if isinstance(bound_max, list):
            bound_max = torch.tensor(bound_max, dtype=torch.float32)    # 如果bound_max类型是list，将其转换为PyTorch张量
        if isinstance(bound_min, np.ndarray):
            bound_min = torch.from_numpy(bound_min).float() # 如果bound_min类型是numpy.ndarray(numpy数组类型)，使用torch.from_numpy将其转换为PyTorch张量
        if isinstance(bound_max, np.ndarray):
            bound_max = torch.from_numpy(bound_max).float() # 如果bound_min类型是numpy.ndarray(numpy数组类型)，使用torch.from_numpy将其转换为PyTorch张量
        self.bound_min = bound_min - self.threshold # if none : bound_min = -1 else min = bmin - threshold
        self.bound_max = bound_max + self.threshold # if none : bound_max = 1  else max = bmax + threshold

        self.is_cut = is_cut                        # bool，是否裁剪-针对生成的双层网格，当模型过于复杂是取消裁剪（false）
        self.region_rate = region_rate              # mini-cut的参数

        self.use_exist_mesh = use_exist_mesh        # 使用现有的 marching-cub生成的网格
        if self.use_exist_mesh == 1:                # ？
            self.use_exist_mesh=True
        else:
            self.use_exist_mesh= False

        self.export_grad_field = export_grad_field  # 是否计算梯度？
        if self.export_grad_field == 1:             # ？
            self.export_grad_field=True
        else:
            self.export_grad_field= False

        self.optimizer = None                       # 这里是什么的优化器？
        self.query_func = query_func                # ？


    def evaluate(self):

        # os.makedirs(self.out_dir, exist_ok=True) # 创建一个递归目录，若存在则继续向下执行代码
        query_func = self.query_func
        device = torch.cuda.current_device()
        
        print(f"当前使用的GPU编号:{self.local_weight}")
        print(f"当前使用的GPU编号:{device}")
        # if (not self.use_exist_mesh) or ( not os.path.exists(self.out_dir + '/MC_mesh.ply')):
        if True:
            # 判断 在输出文件目录下，是否存在MC_mesh.ply文件 同时 是否使用现有模型
            # 若 不存在这个文件，且使用该模型，则打印
            # if not os.path.exists(self.out_dir + '/MC_mesh.ply') and self.use_exist_mesh:
            #     print("No existing MC mesh, generating ...")
            # 提取具有分辨路的网格： 256^3/512^3 (模型分辨率)
            print('Extracting mesh with resolution: {}'.format(self.resolution))
            # 创建新模型，u是一个正常结果，g是一个求导结果
            u, g = extract_fields(self.bound_min, self.bound_max, self.resolution, query_func,self.export_grad_field)
            # np.save('/pub/data/yufg/data/cloth_npz/{self.dataname}', u = u, g=g)

            # np.savez(os.path.join(self.base_exp_dir, "field_{}.npz".format(str(self.threshold))),
            #           df = u, grad=g,bound_min=self.bound_min.cpu().numpy(),bound_max=self.bound_max.cpu().numpy())
            
            u_sclice = u[u.shape[0]//2]
            u_sclice = u_sclice/u_sclice.max()
            self.mesh = threshold_MC(u, self.threshold, self.resolution, bound_min=self.bound_min, bound_max=self.bound_max)
        else:
            self.mesh = trimesh.load_mesh(self.out_dir + '/MC_mesh.ply')
            print("mesh")
        

        # self.mesh.export(self.out_dir + '/MC_mesh.ply')



        ############ 初始化 ##############
        xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda() # 顶点值赋给变量xyz，(n,3)float
        xyz.requires_grad = True # 标记为需要计算梯度，在反向传播时会计算相对于xyz的梯度
        self.optimizer = VectorAdam([xyz]) # 创建了一个优化器，将xyz添加为需要优化的参数，用于更新xyz的数值以最小化损失函数
        laplacian_op = laplacian_calculation(self.mesh).cuda() # 计算，三角形网格相关的拉普拉斯算子，将其移动到gpu上计算，
        vertex_faces = np.asarray(self.mesh.vertex_faces)   # 将self.mesh.vertex_faces转换为numpy数组，并将结果保存在vertex_faces中，其中vertex_faces表示每个顶点所关联的面片
        face_mask = np.ones_like(vertex_faces).astype(bool) # 创建一个bool类型元素组成的数组，face_mask，形状与vertex_faces相同，且值为true
        face_mask[vertex_faces==-1] = False                 # 将vertex_faces中等于-1的元素对应的face_mask元素值设为False，/（可能表示某些顶点不存在关联的面片？）
        
        faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()
        vertices_loss = torch.zeros((len(self.mesh.vertices),1)).cuda() #yfg
        vertices_loss.requires_grad = False
        faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()
        local_process = False   # 是否正在局部处理
        selected_vertex_indices = torch.arange(self.mesh.vertices.shape[0]) # 选中的顶点/ 在全局处理时，选择所有顶点，在选择局部处理时，选择部分顶点
        selected_faces_indices = torch.arange(self.mesh.faces.shape[0]) # 选中的顶点/ 在全局处理时，选择所有面片，在选择局部处理时，选择部分面片
        # op_xyz = xyz
        # 开始进行迭代：最大迭代 max_iter 轮
        for it in range(self.max_iter):


            if it == 20:
                faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()
            
            # if it % 50 == 0:
                # heatmap_mesh = self.mesh.copy()
                # faces_loss_max = torch.max(faces_loss)
                # # faces_loss_normalized = (faces_loss- torch.min(faces_loss)) / (torch.max(faces_loss) - torch.min(faces_loss))
                # # faces_loss_normalized = faces_loss_normalized.cpu().numpy() * 5
                # faces_loss_normalized = ((faces_loss / faces_loss_max)).cpu().numpy()
                # faces_loss_normalized = faces_loss_normalized.clip(0,10)
                # faces_loss_normalized = np.append(faces_loss_normalized, 0)
                # faces_loss_normalized = np.append(faces_loss_normalized, 10)
                # color_map = matplotlib.colormaps["jet"]
                # visual_color = np.zeros((heatmap_mesh.faces.shape[0]))
                # visual_color = color_map(faces_loss_normalized)
                # visual_color = visual_color[:-2]
                # visual_color = visual_color.reshape(visual_color.shape[0], 4)

                # visuals = trimesh.visual.color.ColorVisuals(vertex_colors=None, face_colors=visual_color)

                # print("visual_color:",visual_color.shape)
                # heatmap_mesh.visual = visuals
                # heatmap_mesh.export(self.out_dir + '/{}_heatmap_{}_{}.obj'.format(self.dataname,it,str(self.threshold)))

            if it % 50 == 0 and it > 0 and it <= 1000:
                self.mesh.vertices = xyz.detach().cpu().numpy()
                faces_loss_numpy = faces_loss.cpu().numpy()
                faces_loss_numpy = np.squeeze(faces_loss_numpy)
                # np.savez(self.out_dir + '/loss_data_{}.npz'.format(it), faces_loss=faces_loss_numpy)
                updata, new_face_mask = self.subvision(faces_loss = faces_loss_numpy, it = it, weight = self.sub_weight)                    


                if updata:
                    local_process = False
                    # self.mesh.export(self.out_dir + '/{}_subdivision_{}_{}.ply'.format(self.dataname,it,str(self.threshold)))
                    xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
                    xyz.requires_grad = True
                    faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()

                    self.optimizer.param_groups.clear()
                    self.optimizer.state.clear()
                    self.optimizer.add_param_group({'params': [xyz]})

                    vertices_loss = torch.zeros((xyz.shape[0],1)).cuda() #yfg
                    faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()
                    laplacian_op = laplacian_calculation(self.mesh).cuda()

                    vertex_faces = np.asarray(self.mesh.vertex_faces)
                    face_mask = np.ones_like(vertex_faces).astype(bool)
                    face_mask[vertex_faces == -1] = False          

                    selected_map = torch.full((self.mesh.vertices.shape[0],), -1, dtype=torch.int64)
                    selected_vertex_indices = torch.arange(self.mesh.vertices.shape[0])
                    selected_faces_indices = torch.arange(self.mesh.faces.shape[0]) # 选中的顶点/ 在全局处理时，选择所有面片，在选择局部处理时，选择部分面片
            # 局部处理
            if it == self.local_it:
                # self.mesh.export(self.out_dir + '/{}_jubu_{}_{}.ply'.format(self.dataname,it,str(self.threshold)))
                local_process = True
                select_face_mask = torch.zeros(len(faces), dtype=torch.bool)
                mask = torch.zeros(len(faces), dtype=torch.bool)
                select_face_loss=torch.squeeze(faces_loss)
                select_face_mask[select_face_loss>(self.local_weight * select_face_loss.mean())] = True    # 将面积大于 subdiweight * 平均值 的三角形设置为true
                
                select_faces_subset = faces[select_face_mask]                           # 面片（n,3）
                selected_vertex_indices = torch.unique(select_faces_subset.flatten())   # 获取选中的顶点的索引
                selected_faces_indices = torch.nonzero(select_face_mask).squeeze()      # 获取选中的面片的索引

                vertex_referenced = torch.zeros(len(self.mesh.vertices), dtype=bool)
                vertex_referenced[select_faces_subset] = True
                vertex_inverse = torch.zeros(len(self.mesh.vertices), dtype=torch.int64)
                vertex_inverse[vertex_referenced] = torch.arange(vertex_referenced.sum())
                selected_faces = vertex_inverse[select_faces_subset]

                mesh = self.mesh
                initial_color = [200, 200, 200, 0]  # 设置初始颜色，这里以白色为例
                mesh.visual.face_colors = initial_color
                color = np.random.randint(0, 256, size=3).tolist() + [255]  # RGB + Alpha
                mesh.visual.face_colors[selected_faces_indices.cpu().numpy()] = color
                # mesh.export(self.out_dir + '/{}_jubu_selected_{}_{}.ply'.format(self.dataname,it,str(self.threshold)))
                laplacian_op = laplacian_calculation(mesh = self.mesh,equal_weight = True,selected = selected_vertex_indices.cpu().numpy()).cuda()
                selected_vertex_indices_np = selected_vertex_indices.cpu().numpy()
                selected_xyz = torch.from_numpy(self.mesh.vertices[selected_vertex_indices_np].astype(np.float32)).cuda()
                selected_xyz.requires_grad = True
                self.optimizer.param_groups.clear()
                self.optimizer.state.clear()
                self.optimizer.add_param_group({'params': [selected_xyz]})
                vertices_loss = torch.zeros((selected_xyz.shape[0],1)).cuda() #yfg
                faces_loss = torch.zeros((len(selected_faces_indices),1)).cuda()
                xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
                faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()
            # 切割
            if it == 9000:
                self.mesh.vertices = xyz.detach().cpu().numpy()
                # self.mesh.export(self.out_dir + '/{}_DBScan_forward.ply'.format(self.dataname))
                faces_loss_numpy = faces_loss.cpu().numpy()
                faces_loss_numpy = np.squeeze(faces_loss_numpy)
                # cluster, num, masks = self.DBScan(faces_loss = faces_loss_numpy, weight = 3)
                num = self.DBScan(faces_loss = faces_loss_numpy, weight = 3)
                if num > 0:
                    # self.ModifyTopo(cluster)
                    # self.ModifyTopo_test(masks)
                    # self.mesh.export(self.out_dir + '/{}_cluster_{}_{}.ply'.format(self.dataname,it,str(self.threshold)))
                    xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).cuda()
                    xyz.requires_grad = True
                    # set optimizer to xyz
                    faces = torch.tensor(self.mesh.faces, dtype=torch.int64).cuda()
                    
                    self.optimizer.param_groups.clear()
                    self.optimizer.state.clear()
                    self.optimizer.add_param_group({'params': [xyz]})
                    vertices_loss = torch.zeros((xyz.shape[0],1)).cuda() #yfg
                    faces_loss = torch.zeros((len(self.mesh.faces),1)).cuda()

                    laplacian_op = laplacian_calculation(self.mesh).cuda()
                    vertex_faces = np.asarray(self.mesh.vertex_faces)
                    face_mask = np.ones_like(vertex_faces).astype(bool)
                    face_mask[vertex_faces == -1] = False
                    selected_vertex_indices = torch.arange(self.mesh.vertices.shape[0])
                    selected_faces_indices = torch.arange(self.mesh.faces.shape[0]) # 选中的顶点/ 在全局处理时，选择所有面片，在选择局部处理时，选择部分面片                
            
            self.update_learning_rate(it)   # 更新学习率。
            epoch_loss = 0  # 初始化损失函数
            self.optimizer.zero_grad()  # 初始化，将所有参数的梯度设置为 0
            
            if local_process:
                vloss = torch.zeros((selected_xyz.shape[0],1)).cuda() #yfg
                num_samples = selected_xyz.shape[0]
            else:
                vloss = torch.zeros((xyz.shape[0],1)).cuda() #yfg
                num_samples = xyz.shape[0]  # 顶点个数 
            head = 0    # 从第 0 个节点开始
            # 分批处理所有的点和面，通过每次取一部分子集的方式。
            while head< num_samples:
                # # 从 head 开始 到 head + max_batch 或是 顶点上限 结束，选择一部分顶点子集进行计算
                if local_process:
                    sample_subset = selected_xyz[head: min(head + self.max_batch, num_samples)]
                else:
                    sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)  # 取出顶点子集的距离
                
                vertices_tmp = df.detach().clone()
                vloss[head: min(head + self.max_batch, num_samples)] += vertices_tmp
                # vertices_loss[head: min(head + self.max_batch, num_samples)] += vertices_tmp
                if vertices_loss.mean() != 0:
                    vertices_l = vertices_loss[head: min(head + self.max_batch, num_samples)]
                    # df = df * ( vertices_l / (vertices_loss.mean()) )#?
                    ratio = vertices_l / vertices_loss.mean()
                    df = df * (ratio.clip(1, 10))
                    # print(torch.max(vertices_l / (vertices_loss.mean())))
                df_loss = df.mean()
                loss = df_loss  # 当前样本子集的损失值

                s_value = calculate_s(xyz, self.mesh.faces) # 每个三角形的面积
                face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]
                face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                face_weight = torch.sum(face_weight, dim=1)
                face_weight = torch.sqrt(face_weight.detach())
                face_weight = face_weight.max() / face_weight
                if local_process:
                    lap = laplacian_step(laplacian_op, xyz, selected=selected_xyz)
                else:
                    lap = laplacian_step(laplacian_op, xyz)
                lap_v = torch.mul(lap, lap)
                lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
                laplacian_loss = face_weight.detach() * torch.sum(lap_v, dim=1)
                laplacian_loss_mean = laplacian_loss.mean()
                # laplacian_weight1 = 170 * torch.exp(torch.tensor(0.0) - (it / (self.max_iter)))
                # laplacian_loss = laplacian_weight1 * self.laplacian_weight * laplacian_loss_mean
                laplacian_loss = 1800 * laplacian_loss_mean
                loss = loss + laplacian_loss # 与当前样本的损失值 相加得到新的损失
                ###############################################################
                # edges_unique_length = np.array(self.mesh.edges_unique_length)
                # faces_unique_edges = np.array(self.mesh.faces_unique_edges)
                # faces_perimeter = np.sum(edges_unique_length[faces_unique_edges], axis=1)
                # faces_perimeter_weight = torch.from_numpy(faces_perimeter[vertex_faces[head: min(head + self.max_batch, num_samples)]]).cuda()
                # faces_perimeter_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                # faces_perimeter_weight = torch.sum(faces_perimeter_weight, dim=1)
                # faces_perimeter_weight = torch.sqrt(faces_perimeter_weight.detach())
                # faces_perimeter_weight = (faces_perimeter_weight.max() / faces_perimeter_weight)
                # lap_p = torch.mul(lap, lap)
                # lap_p = lap_p[head: min(head + self.max_batch, num_samples)]
                # faces_perimeter_loss = faces_perimeter_weight.detach() * torch.sum(lap_p, dim=1)
                # faces_perimeter_loss_mean = faces_perimeter_loss.mean()
                # laplacian_weight2 = 18 * torch.exp(2 * torch.tensor(it,dtype=torch.float64) / self.max_iter) + 50
                # faces_perimeter_loss =laplacian_weight2 * self.laplacian_weight * faces_perimeter_loss_mean
                # loss = loss + faces_perimeter_loss # 与当前样本的损失值 相加得到新的损失值
                ###############################################################
                epoch_loss += loss.data # 对每一轮的损失值进行叠加 / 其中loss.data和loss 的区别是，loss.data只提供损失值的张量，而loss是包含损失值和梯度信息的张量。
                loss.backward()         # 用于计算损失，以使其向减小损失的方向移动
                head += self.max_batch  # 更新子集
            vertices_loss += vloss
            
            if local_process:
                floss = torch.zeros((len(selected_faces_indices),1)).cuda()
                mid_num_samples = len(selected_faces_indices)  # 面片的总数
            else:
                floss = torch.zeros((len(self.mesh.faces),1)).cuda()
                mid_num_samples = len(self.mesh.faces)  # 面片的总数
            mid_head = 0                            # 从第 0 个面片开始
            while mid_head< mid_num_samples:        
                if local_process:
                    mid_points = get_mid(selected_xyz, selected_faces)  # 输入 点 和 面 ，返回三角形面片的中点。
                else:
                    mid_points = get_mid(xyz, faces)  # 输入 点 和 面 ，返回三角形面片的中点。
                sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]  # 选出子集
                mid_df = query_func(sub_mid_points) # 查询三角形面片的中点坐标处 的 距离
                faces_tmp = mid_df.detach().clone() 
                floss[mid_head: min(mid_head + self.max_batch, mid_num_samples)] += faces_tmp
                # faces_loss[mid_head: min(mid_head + self.max_batch, mid_num_samples)] += faces_tmp
                if faces_loss.mean() != 0:
                    faces_l = faces_loss[mid_head: min(mid_head + self.max_batch, mid_num_samples)] # 取出子集损失值
                    # mid_df = mid_df * (faces_l / faces_loss.mean())
                    ratio = faces_l / faces_loss.mean()
                    mid_df = mid_df * (ratio.clip(1, 10))
                    # print(torch.max(faces_l / faces_loss.mean()))
                mid_df_loss = mid_df.mean() # 求到这些距离的平均值，作为中点数据的损失值
                loss = mid_df_loss  # 为中点数据的损失函数值赋给总体损失函数loss
                epoch_loss += loss.data     # 对每一轮的损失值进行叠加 / 其中loss.data和loss 的区别是，loss.data只提供损失值的张量，而loss是包含损失值和梯度信息的张量。    
                loss.backward()             # 用于计算损失，以使其向减小损失的方向移动
                mid_head += self.max_batch  # 更新子集
            faces_loss += floss

            f_loss = faces_loss.detach().repeat_interleave(3).cuda()    # [n,] -> [3n,]
            v_loss = torch.zeros(len(self.mesh.vertices)).cuda()        # 
            v_loss.scatter_add_(0,faces[selected_faces_indices].flatten(),f_loss)   # [n,3] -> [3n,] 与f_loss一一对应，再将对应的值添加到对应的顶点中
            v_loss = (v_loss - torch.min(v_loss)) / (torch.max(v_loss) - torch.min(v_loss))
            N = copy.deepcopy(self.mesh.vertex_normals)
            N = torch.tensor(N).cuda()
            if local_process:
                N = N[selected_vertex_indices]
                v_loss = v_loss[selected_vertex_indices]
            # lamta = 0.0001 - (it/self.max_iter)*(it/self.max_iter - 1)
            self.optimizer.step(N, v_loss)   # 根据计算梯度，更新模型的参数。
            # self.optimizer.step()

            if (it+1) % self.report_freq == 0:
                print(" {} iteration, loss={}".format(it, epoch_loss))

            if (it+1) % self.save_iter == 0:
                if local_process:
                    xyz[selected_vertex_indices] = selected_xyz.detach()
                    points = xyz.detach().cpu().numpy()
                else:
                    points = xyz.detach().cpu().numpy()
                self.mesh.vertices = points
            if (it%100==0):
                self.mesh.export( './{}_{}_Optimize_{}.ply'.format("test",it,str(self.threshold)))
                # self.mesh.export(self.out_dir + '/{}_{}_Optimize_{}.ply'.format(self.dataname,it,str(self.threshold)))

        final_mesh = trimesh.Trimesh(vertices=xyz.detach().cpu().numpy(), faces=self.mesh.faces, process=False)
        
        if self.is_cut == 1:
            # 导入 cut_mesh_v2 函数
            from cut_v2 import cut_mesh_v2
            # 调用 cut_mesh_v2 函数 对最终网格进行切割，传入最终网格对象 和 self.region_rate
            final_mesh_cuple = cut_mesh_v2(final_mesh,self.region_rate)
            # 如果切割成功
            if final_mesh_cuple is not None:
                # 将骑个猴的两个网格对象分别导出到 指定的输出目录，文件名包含，原始数据名称、索引等信息。
                final_mesh_1 = final_mesh_cuple[0]
                final_mesh_2 = final_mesh_cuple[1]
                # final_mesh_1.export(self.out_dir + '/' + '{}-0_{}_new.ply'.format(self.dataname,str(self.threshold)))
                # final_mesh_2.export(self.out_dir + '/' + '{}-1_{}_new.ply'.format(self.dataname,str(self.threshold)))
                # 分别计算两个切割后网格的损失值，通过调用 compute_df 方法
                loss1 = self.compute_df(final_mesh_1)
                loss2 = self.compute_df(final_mesh_2)
                # 选择损失值较小的 网络， 作为最终网络
                if loss1<loss2:
                    final_mesh = final_mesh_1
                else:
                    final_mesh = final_mesh_2
                # 将最终的网格 导出到指定的输出目录
                # print("exported result")
                # final_mesh.export(self.out_dir + '/' + '{}_{}_new.ply'.format(self.dataname,str(self.threshold)))
            else:
                print("It seems that model is too complex, cutting failed. Or just rerunning to try again.")
        # else:
            # final_mesh.export(self.out_dir + '/' + '{}_{}_new.ply'.format(self.dataname,str(self.threshold)))
        return final_mesh,final_mesh,final_mesh

    def op_weight(self, loss):

        a = torch.max(loss)
        b = a / 2

        condition = loss < b
        result = torch.zeros_like(loss)
        
        result[condition] = (a / b) * loss[condition]
        result[~condition] = (a / (b - a)) * loss[~condition] + (a ** 2 / (a - b))
        
        return result
    
    def update_learning_rate(self, iter_step):
        warn_up = self.warm_up_end      # 让学习率逐渐增大的一个参数
        max_iter = self.max_iter        # 最大迭代次数
        init_lr = self.learning_rate    # 当前的学习率

        # 如果 当前 迭代次数 < warn_up
        # lr 随着迭代次数的增大而减小 lr = iter_step / warn_up，线性增加
        # 反之 当前 迭代次数 > warn_up
        # lr 使用 余弦退火？的方式:https://blog.csdn.net/weixin_35848967/article/details/108493217
        # lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        # 更好
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/((max_iter - warn_up)) * math.pi) + 1)
        # lr 先是减小，直到iter_step == warn_up ， 然后 先缓慢减小，再快速减小，再缓慢减小（）。
    
        
        lr =lr * init_lr
        # 当 当前 迭代次数 超过二百轮的时候，让学习率加速减小。
        if iter_step>=200:
            lr *= 0.1
        # 将结果送入迭代器，VectorAdam 中
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
        # pc.export(self.out_dir + '/' + '{}_pc.ply'.format(self.dataname))

    def compute_df(self,mesh):
    
        # 对 mesh 随即采样 十万个点，转换为 tensor 类型， 并在gpu上计算，采样后赋值给xyz
        xyz = torch.from_numpy(mesh.sample(100000).astype(np.float32)).cuda()
        # 记录该 点集 的梯度信息
        xyz.requires_grad = True
        # 获取采样点的总数，并初始化 头指针 和 损失函数
        num_samples = xyz.shape[0]
        head = 0
        loss = 0
        while head < num_samples:
            # 取出部分子集
            sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
            # 查询距离
            df = self.query_func(sample_subset)
            # 距离合
            df_loss = df.sum().data
            # print("cal complete")

            # if self.use_edge_loss:
            #     edg_loss = cir_loss(xyz, origin_edg,faces=mesh.faces)
            # else:
            #     edg_loss = torch.FloatTensor(0)
            # print(laplacian_loss)
            # w = 0.5 * (math.cos((it) / (self.max_iter) * math.pi/2) + 0.2)
            # 距离损失函数 累加在 损失函数中
            loss += df_loss
            head += self.max_batch
        # 求出每个点的平均损失值。
        return loss/num_samples

    def subvision(self, faces_loss, it,sub_threshould=0.001, weight = 4):

        def calcul_normal(a,b,c):
            return np.cross(b - a, c - a)
        
        def res_normal(or_normal, trimesh_normal, a, b, c):
            dot_p1 = np.dot(or_normal, trimesh_normal)
            if dot_p1 >= 0:
                self.refine_faces.append([a, b, c])
            else:
                self.refine_faces.append([c, b, a])
        
        faces = np.asarray(self.mesh.faces) # asarray 没有创建输入对象的副本，而是共享相同的数据。

        vertices = np.asarray(self.mesh.vertices)   # 面和点 都是会对原始数据进行修改。
        areas = calculate_s(torch.from_numpy(vertices),faces).numpy()   # 求出三角形的面积
        face_mask = np.zeros(len(faces), dtype=bool)    # 创建一维的数组，全部为false，(n,) n为三角形个数

        # print("areas:", type(areas))
        # print("faces_loss:", type(faces_loss))

        face_mask[(faces_loss>(weight * faces_loss.mean())) & (areas>1.5 * areas.mean())] = True    # 将面积大于 subdiweight * 平均值 的三角形设置为true

        if not any(face_mask):
            return False, face_mask

        # the (c, 3) int array of vertex indices
        faces_subset = faces[face_mask] # 将 设置为 true 的三角形保存下来，其[m,3] m 是true的三角形个数，3 是三角形的三个顶点。
        faces_subset_loss = faces_loss[face_mask]
        # sub_weights = 2*weights[face_mask]
        # sub_weights = np.repeat(sub_weights, 4)
        # find the unique edges of our faces subset
        # geometry是 trimesh库的一个模块，提供了一些用于几何计算和处理的函数和类
        #edges = np.sort(trimesh.geometry.faces_to_edges(faces_subset), axis=1)  # faces_to_edges：将面数转换为边数组[3n,2] n 面的数量，2 两个顶点 再进行排序，两个顶点从小到大
        edges = np.sort(trimesh.geometry.faces_to_edges(faces_subset), axis=1)
        # 去除edges中重复的行，[n - m],n条边，m条重复的边。
        # inverse可以利用唯一行的索引恢复edges。
        # array:[[1,2],[3,4],[1,2],[2,3]]  /  unique:[0,3,1]   /  inverse:[0,2,0,1] / array[inverse] 得到原始的边
        unique, inverse = trimesh.grouping.unique_rows(edges)   
        # then only produce one midpoint per unique edge
        mid = vertices[edges[unique]].mean(axis=1)  # 计算每条每一边的中点坐标，mean(axis = 1)是计算边的中点坐标
        mid_idx = inverse.reshape((-1, 3)) + len(vertices)  # inverse 从一维转换为二维，(3n) -> (n,3) n为三角形个数，3为其三条边。每个唯一边对应的面索引，这个 +len(vertices) 表示点的索引
        # 然后通过 得到的索引 用 mesh.face_adjacency 查找， 得到的是 公用一条边的面。
        # 一条边得到两个面。
        face_adj2sub = self.mesh.face_adjacency[self.mesh.face_adjacency_edges_tree.query(edges)[1]].flatten()  # 得到给定边相邻的面
        refine_face_idxs = np.zeros(len(faces), dtype=int)  

        # 比上面的循环快一倍左右
        # refine_face_edges = [[] for _ in range(len(faces))]            # 预先分配面的边缘信息
        # refine_face_unshared = [[] for _ in range(len(faces))]         # 预先分配面未共享边的信息
        # refine_face_mid_idxs = [[] for _ in range(len(faces))]         # 预先分配面的中心点索引信息
        ########################### 改动 #########################################
        # # 指一条边，在共享的平面中，返回两个不在共享边上的两个顶点的顶点索引值 / [m,2] -> [2n] , n 条共享边
        # face_adjacency_unshared = self.mesh.face_adjacency_unshared[self.mesh.face_adjacency_edges_tree.query(edges)[1]].flatten()
        # ################################# test #################################

        # mask_edges = np.repeat(face_mask, 3)
        # all_edges = np.sort(trimesh.geometry.faces_to_edges(faces), axis=1)
        # unique_edges, idx, counts = np.unique(all_edges, axis=0, return_index=True, return_counts=True)

        # referenced = np.zeros(len(all_edges), dtype=bool)
        # referenced[idx] = True
        # inverse_idx = np.zeros(len(all_edges), dtype=np.int64)
        # inverse_idx[referenced] = np.arange(referenced.sum())
        # mask_edges_idx = np.ones(len(all_edges), dtype=bool)
        # mask_edges_idx[idx[counts == 1]] = False

        # edge_idx = np.arange(len(all_edges))[mask_edges == True]
        # ################################# over #################################
        # # face_adj2sub: 2n， n条共享边，一条共享边有两个面， 有两个不在共享边上的顶点。
        # for i, f_idx in enumerate(face_adj2sub):
        #     # ？
        #     if mask_edges_idx[edge_idx[i//2]] == True:
        #         refine_face_mid_idxs[f_idx].append(inverse[i//2])   # 每个面， 共享的是哪一条边, 使用的是边索引
        #         refine_face_edges[f_idx].append(edges[i // 2])  # 每个面，共享的是哪一条边， 使用的是边的两个顶点，两个顶点是顶点索引。

        #         refine_face_idxs[f_idx] += 1    # f_idx 下有几个相邻面，例如 A 面中有三个相邻面
        #         refine_face_unshared[f_idx].append(face_adjacency_unshared[i])  # 把第 i 个值（第i个相邻面中不相邻的顶点） 添加到第 f_idx 个面中(f_idx为面的索引值)
        
        
        refine_face_edges = []                  # 面的边缘信息
        for i in range(len(faces)):
            refine_face_edges.append([])

        refine_face_unshared = []               # 面未共享边的信息
        for i in range(len(faces)):
            refine_face_unshared.append([])

        refine_face_mid_idxs = []               # 面的中心点索引信息
        for i in range(len(faces)):
            refine_face_mid_idxs.append([])
        refine_face_weight=[]                   # 面权重信息

        face_adjacency_unshared = self.mesh.face_adjacency_unshared[self.mesh.face_adjacency_edges_tree.query(edges)[1]].flatten()
        for i, f_idx in enumerate(face_adj2sub):
            refine_face_mid_idxs[f_idx].append(inverse[i//2])
            refine_face_edges[f_idx].append(edges[i // 2])
            refine_face_idxs[f_idx] += 1
            refine_face_unshared[f_idx].append(face_adjacency_unshared[i])   

        refine_face_idxs[face_mask] = 0 # 大于0.8的 归零
        # the new faces_subset with correct winding
        refine_0 = np.where(refine_face_idxs == 0)[0]   #
        if(refine_face_idxs > 3).any():
                print("???") 
        # print("refine_face_idxs:",refine_face_idxs)
        refine_1 = np.where(refine_face_idxs == 1)[0]   # 找到有一个相邻面的 面的索引
        refine_2 = np.where(refine_face_idxs == 2)[0]   # 找到有两个相邻面的 面的索引
        refine_3 = np.where(refine_face_idxs == 3)[0]   # 找到有三个相邻面的 面的索引
        self.refine_faces = []
        self.refine_loss = []
        face_normals = self.mesh.face_normals

        # id 是面的索引值
        # 新建的两个三角形因该如何考虑上下
        for id in refine_1:
            if face_mask[id]:
                print("???")
            face_mask[id] = True    # 将相邻面也置为 true
            id_loss = faces_loss[id]    # 获取该面片下的loss值
            self.refine_loss.append(id_loss)    # 一个面片下生成两个新面片
            self.refine_loss.append(id_loss)
            # refine_face_weight.append(2*weights[id])
            # refine_face_weight.append(2 * weights[id])
            
            mid_point_1_idx = refine_face_mid_idxs[id][0] + len(vertices)   # 因为只有一个相邻的边，这条边的索引值，令其加上当前所有点的数量，作为这条边的中间点的索引值。
            p1,p2 = refine_face_edges[id][0]    # 获取这条边上的两个点
            p3 = refine_face_unshared[id][0]    # 获取这个三角形上不在那条边上的点。
            p1_ver = vertices[p1]
            p2_ver = vertices[p2]
            p3_ver = vertices[p3]

            or_normal = face_normals[id]
            mid_point_1_idx_ver = mid[mid_point_1_idx - len(vertices)]

            trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, p3_ver)
            trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, p3_ver)
            res_normal(or_normal, trimesh1_normal, p1, mid_point_1_idx, p3)
            res_normal(or_normal, trimesh2_normal, mid_point_1_idx, p2, p3)

        for id in refine_2:
            if face_mask[id]:
                print("???")
            face_mask[id] = True
            id_loss = faces_loss[id]    # 获取该面片下的loss值
            self.refine_loss.append(id_loss)    # 一个面片细分成三个新面片
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            # refine_face_weight.append(2 * weights[id])
            # refine_face_weight.append(2 * weights[id])
            # refine_face_weight.append(2 * weights[id])
            mid_point_1_idx = refine_face_mid_idxs[id][0] + len(vertices)   
            mid_point_2_idx = refine_face_mid_idxs[id][1] + len(vertices)
            p1,p2 = refine_face_edges[id][0]
            p3 = refine_face_unshared[id][0]
            p4 = refine_face_unshared[id][1]
            # mid_2 in edge(p2,p3)

            p1_ver = vertices[p1]
            p2_ver = vertices[p2]
            p3_ver = vertices[p3]

            or_normal = face_normals[id]

            mid_point_1_idx_ver = mid[mid_point_1_idx - len(vertices)]
            mid_point_2_idx_ver = mid[mid_point_2_idx - len(vertices)]

            if p4 == p1:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, p3_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, mid_point_2_idx_ver, p3_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, mid_point_2_idx_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, p3)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, mid_point_2_idx, p3)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p2, mid_point_2_idx)
            else:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, mid_point_2_idx_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p3_ver, mid_point_2_idx_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, p3_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, mid_point_2_idx)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p3,  mid_point_2_idx)
                res_normal(or_normal, trimesh3_normal,mid_point_1_idx, p2, p3)

        for id in refine_3:
            if face_mask[id]:
                print("???")
            face_mask[id] = True
            id_loss = faces_loss[id]    # 获取该面片下的loss值
            self.refine_loss.append(id_loss)    # 一个面片下细分成四个新面片
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            self.refine_loss.append(id_loss)
            mid_point_1_idx = refine_face_mid_idxs[id][0] + len(vertices)
            mid_point_2_idx = refine_face_mid_idxs[id][1] + len(vertices)
            mid_point_3_idx = refine_face_mid_idxs[id][2] + len(vertices)
            p1,p2 = refine_face_edges[id][0]
            p3 = refine_face_unshared[id][0]
            p4 = refine_face_unshared[id][1]            
            p5 = refine_face_unshared[id][2]

            p1_ver = vertices[p1]
            p2_ver = vertices[p2]
            p3_ver = vertices[p3]

            or_normal = face_normals[id]

            mid_point_1_idx_ver = mid[mid_point_1_idx - len(vertices)]
            mid_point_2_idx_ver = mid[mid_point_2_idx - len(vertices)]
            mid_point_3_idx_ver = mid[mid_point_3_idx - len(vertices)]
            # yfg: 沿着 p1 -> p2 -> p3  if ① else ②
            if p1 == p5:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, mid_point_2_idx_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, mid_point_3_idx_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, mid_point_3_idx_ver, mid_point_2_idx_ver)
                trimesh4_normal = calcul_normal(mid_point_2_idx_ver, mid_point_3_idx_ver, p3_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, mid_point_2_idx)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p2, mid_point_3_idx)
                res_normal(or_normal, trimesh3_normal,mid_point_1_idx,mid_point_3_idx,mid_point_2_idx)
                res_normal(or_normal, trimesh4_normal,mid_point_2_idx, mid_point_3_idx, p3)
            else:
                trimesh1_normal = calcul_normal(p1_ver, mid_point_1_idx_ver, mid_point_3_idx_ver)
                trimesh2_normal = calcul_normal(mid_point_1_idx_ver, p2_ver, mid_point_2_idx_ver)
                trimesh3_normal = calcul_normal(mid_point_1_idx_ver, mid_point_2_idx_ver, mid_point_3_idx_ver)
                trimesh4_normal = calcul_normal(mid_point_3_idx_ver, mid_point_2_idx_ver, p3_ver)

                res_normal(or_normal, trimesh1_normal,p1, mid_point_1_idx, mid_point_3_idx)
                res_normal(or_normal, trimesh2_normal,mid_point_1_idx, p2, mid_point_2_idx)
                res_normal(or_normal, trimesh3_normal,mid_point_1_idx,mid_point_2_idx,mid_point_3_idx)
                res_normal(or_normal, trimesh4_normal,mid_point_3_idx, mid_point_2_idx, p3)

        #refine_face_weight = np.array(refine_face_weight)
        # faces_subset: 三角形为true的， [m,3]
        # mid_idx: [n,3] n 表示n个三角形， 3 是其三条边， 每条边加上了len[vertices]，表示每个边都从中间生成了一个新的节点，每个点的新的索引值。
        # 上面构建的是 与较大面积相邻的三角形， 下面构建的是 较大面积的三角形，从一个 变成四个。

        f = np.column_stack([faces_subset[:, 0],
                             mid_idx[:, 0],
                             mid_idx[:, 2],

                             mid_idx[:, 0],
                             faces_subset[:, 1],
                             mid_idx[:, 1],
                             
                             mid_idx[:, 2],
                             mid_idx[:, 1],
                             faces_subset[:, 2],
                             
                             mid_idx[:, 0],
                             mid_idx[:, 1],
                             mid_idx[:, 2]]).reshape((-1, 3))
        f_loss = np.repeat(faces_subset_loss, 4)

        # add the 3 new faces_subset per old face all on the end
        # by putting all the new faces after all the old faces
        # it makes it easier to understand the indexes
        
        refine_faces = np.array(self.refine_faces)
        refine_loss = np.array(self.refine_loss)

        original_colors = self.mesh.visual.face_colors[~face_mask]  # 保留原始网格面中，未被标记的颜色

        new_faces = np.vstack((faces[~face_mask], f))   # 原始面片和新面片
        # new_loss = np.hstack((faces_loss[~face_mask], f_loss))

        # new_loss = np.vstack((faces_loss[~face_mask], f_loss)) # 原始损失累加和新损失累加
        # new_weight = np.vstack((weights[~face_mask], sub_weights))
        new_faces = np.vstack((new_faces, refine_faces))    # 新面片和相邻面片
        new_faces_mask = np.vstack((f, refine_faces))
        # new_loss = np.hstack((new_loss, refine_loss))      # 新损失累加和相邻损失累加
        # new_weight = np.vstack((new_weight, refine_face_weight))

        # stack the new midpoint vertices on the end
        new_vertices = np.vstack((vertices, mid))   # 添加新顶点
        self.mesh = trimesh.Trimesh(new_vertices,new_faces, process=False)  # 生成新的网络模型
        new_color = np.zeros((len(self.mesh.faces), original_colors.shape[1]))  # 生成新的颜色信息

        new_color[:original_colors.shape[0]] = original_colors  # 将原始颜色信息输入保存
        new_color[-(len(f)+len(refine_faces)):] = [0, 200, 0, 255]  # 将发生过变化的面片信息输入保存

        print("subdivision:", (len(new_faces)-len(faces)))   # 新生成的面片信息
        visuals = trimesh.visual.color.ColorVisuals(vertex_colors=None, face_colors=new_color)
        self.mesh.visual = visuals
        # print("sub_visual_color:",new_color.shape)
        
        return True, new_faces_mask

    def floss_to_vloss(self, faces_loss):
        faces = self.mesh.faces
        vertices_loss = torch.zeros(len(self.mesh.vertices))

        for i, f in enumerate(faces):
            loss = faces_loss[i]
            for v in f:
                vertices_loss[v] += loss

    def spectral_clustering(self, faces_loss):
        '''
            将相邻面片归类
            - input: 面片的损失累加
            - output:n个相邻的面片组:faces_class
        '''
        
        def calculate_w_ij(a,b,sigma=1):
            w_ab = np.exp(-np.sum((a-b)**2)/(2*sigma**2))
            return w_ab

        # 计算邻接矩阵
        def Construct_Matrix_W(data,k=5):
            rows = len(data) # 取出数据行数
            W = np.zeros((rows,rows)) # 对矩阵进行初始化：初始化W为rows*rows的方阵
            for i in range(rows): # 遍历行
                for j in range(rows): # 遍历列
                    if(i!=j): # 计算不重复点的距离
                        W[i][j] = calculate_w_ij(data[i],data[j]) # 调用函数计算距离
                t = np.argsort(W[i,:]) # 对W中进行行排序，并提取对应索引
                for x in range(rows-k): # 对W进行处理
                    W[i][t[x]] = 0
            W = (W+W.T)/2 # 主要是想处理可能存在的复数的虚部，都变为实数
            return W

        def Calculate_Matrix_L_sym(W): # 计算标准化的拉普拉斯矩阵
            degreeMatrix = np.sum(W, axis=1) # 按照行对W矩阵进行求和
            L = np.diag(degreeMatrix) - W # 计算对应的对角矩阵减去w
            # 拉普拉斯矩阵标准化，就是选择Ncut切图
            sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5))) # D^(-1/2)
            L_sym = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix) # D^(-1/2) L D^(-1/2)
            return L_sym

        def normalization(matrix): # 归一化
            sum = np.sqrt(np.sum(matrix**2,axis=1,keepdims=True)) # 求数组的正平方根
            nor_matrix = matrix/sum # 求平均
            return nor_matrix
        
        def generate_random_color():
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            a = 255  # 不透明度设置为255，即完全不透明
            return [r, g, b, a]

        mesh = self.mesh.copy()
        face_wait = faces_loss[(faces_loss > faces_loss.mean())].astype(int) # (t,3),t:面索引,3:三个顶点索引。待聚类面片
        points = np.asarray(self.mesh.vertices.astype(np.float32))  # 得到顶点索引对应的坐标
        fvs = points[face_wait]
        origin_points = np.mean(fvs,axis=1) # 得到每个面片的中点
       
        W = Construct_Matrix_W(origin_points) # 输入面片中点，计算邻接矩阵

        L_sym = Calculate_Matrix_L_sym(W) # 依据W计算标准化拉普拉斯矩阵
        lam, H = np.linalg.eig(L_sym) # 特征值分解
        t = np.argsort(lam) # 将lam中的元素进行排序，返回排序后的下标
        H = np.c_[H[:,t[0]],H[:,t[1]]] # 0和1类的两个矩阵按行连接，就是把两矩阵左右相加，要求行数相等。
        H = normalization(H) # 归一化处理

        from sklearn.cluster import KMeans
        n_clusters = 50
        model = KMeans(n_clusters=n_clusters) # 新建50簇的Kmeans模型
        model.fit(H) # 训练
        labels = model.labels_ # 得到聚类后的每组数据对应的标签类型

            
        res = [[] for i in range(n_clusters)]  # 初始化 tp 列表，创建 k 个空列表
        for i in range(len(face_wait)):
            res[labels[i]].append(face_wait[i])

        initial_color = [255, 255, 255, 255]  # 设置初始颜色，这里以白色为例
        mesh.visual.face_colors = initial_color

        cluster_colors = {}  # 用于保存每个簇的颜色

        # res:[n,m,3] n：第n个簇，m：该簇中又m个面片，3：每个面片的三个顶点。
        for label in labels:
            color = cluster_colors[label]  # 获取该簇对应的颜色
            for face in res[label]:
                for vertex in face:
                    mesh.visual.vertex_colors[vertex] = color

        # mesh.export(self.out_dir + '/{}_spectral_clustering.ply'.format(self.dataname))
        print("spectral_clustering_over")

    def DBScan(self, faces_loss, weight = 5):

        # 修改顺序
        def resort(ring_edges):
            ring_edges = torch.stack(ring_edges)
            indices = torch.where(ring_edges[0][0] == ring_edges[1])
            tmp = torch.tensor([ring_edges[0][1], ring_edges[0][0]])
            result = [tmp]
            if len(indices[0]) == 0:
                indices = torch.where(ring_edges[0][1] == ring_edges[1])
                result = [ring_edges[0]]

            # # 遍历数组，调整顺序
            for i in range(1, len(ring_edges)):
                if indices[0][0] == 0:
                    result.append(ring_edges[i])
                    next = ring_edges[i][1]
                else:
                    tmp = torch.tensor([ring_edges[i][1], ring_edges[i][0]])
                    result.append(tmp)
                    next = ring_edges[i][0]
                
                if i+1 >= len(ring_edges):
                    tmp = torch.tensor([result[-1][1],result[0][0]])
                    result.append(tmp)
                else:
                    indices = torch.where(next == ring_edges[i+1])
            result = torch.stack(result)
            return result
        # 聚类的过程
        def getCluster(diff_list,face_neighbor):

            # 聚类结果字典和聚类编号计数器初始化
            cluster = []
            num = 0
            # 对每个点进行聚类
            while diff_list:
                # print("idf:",len(diff_list))
                pos = diff_list.pop(0)
                # print("pos:",pos)
                # 初始化一个聚类列表，并将该点加入其中
                clusterpoint = []
                clusterpoint.append(pos)
                # 获取满足条件的点
                indices = face_neighbor[pos]
                seedlist = []
                for idx in indices:
                    # 既要距离不超阈值， 又要保证所选的点为待处理的点
                    if idx in diff_list:
                        # 同时还要是相邻的面片：
                        seedlist.append(idx)
                        diff_list.remove(idx)

                # 使用 BFS 算法进行聚类
                while seedlist:
                    # 取出邻域内的第一个点
                    p = seedlist.pop(0)
                    # 将该点加入聚类
                    clusterpoint.append(p)
                    indices = face_neighbor[p]
                    for idx in indices:
                        #保证所选的点为待处理的点
                        if idx in diff_list:
                            # print("idx:",idx)
                            # 同时还要是相邻的面片：
                            seedlist.append(idx)
                            diff_list.remove(idx)
                # 将该聚类添加到聚类结果字典中，并增加聚类编号计数器
                if len(clusterpoint) > 20:
                    cluster.append(clusterpoint)   
                    num+=1
            # 将未被访问过的点（噪声点）添加到聚类结果中
            return cluster, num
        # 判断柱形
        def select_ring(edges):
            if edges.shape == 0:
                return 0, []
            ring_edges = []  # 存储环的边
            ring_num = 0
            visited_edges = set()
            
            for i in range(len(edges) - 1):
                # 寻找初始边
                # print(i)
                if tuple(edges[i].tolist()) in visited_edges:
                    i +=1
                    if ring_num > 2:
                        return 3, []
                    continue
                visited_edges.add(tuple(edges[i].tolist()))
                current_ring_edges = []  # 存储当前环的边
                start_vertex = edges[i][0]  # 环头
                current_vertex = edges[i][1]# 当前环节点
                
                while True:
                    if start_vertex == current_vertex:
                        ring_edges.append(current_ring_edges)  # 将当前环的边添加到结果中
                        ring_num += 1
                        break
                    
                    subscript = torch.where(current_vertex == edges)
                    next_vertex = None
                    for s in range(len(subscript[0])):
                        if tuple(edges[subscript[0][s]].tolist()) not in visited_edges:
                            visited_edges.add(tuple(edges[subscript[0][s]].tolist()))
                            current_ring_edges.append(edges[subscript[0][s]])  # 将当前边加入当前环的边列表
                            if subscript[1][s] == 0:
                                next_vertex = edges[subscript[0][s]][1]
                            else:
                                next_vertex = edges[subscript[0][s]][0]
                            break
                    
                    if next_vertex is None:
                        break
                    current_vertex = next_vertex
            return ring_num, ring_edges
        # 缝合空洞
        def fill_hole(edges, vertices, mesh, mask):

            def dot_product(vector1, vector2):
                product = torch.dot(vector1, vector2)

                # 计算向量的长度
                norm_vector1 = torch.norm(vector1)
                norm_vector2 = torch.norm(vector2)

                # 计算两个向量之间的夹角（弧度）
                angle_radians = torch.acos(product / (norm_vector1 * norm_vector2))

                # 将弧度转换为角度
                angle_degrees = angle_radians * 180 / 3.1415926
                return torch.tensor([180 - angle_degrees])

            def verify_normal_vector(mesh, mask, v):

                target_faces1 = []
                target_faces2 = []
                for face_index, face in enumerate(mesh.faces):
                    if mask[face_index] and v[0] in face and v[1] in face :
                        target_faces1.append(face_index)
                    if mask[face_index] and v[1] in face and v[2] in face :
                        target_faces2.append(face_index)

                faces_normal1 = mesh.face_normals[target_faces1]
                faces_normal2 = mesh.face_normals[target_faces2]
                # 相邻的两个面片的法向量不同是怎么回事？？
                target_v1 = np.squeeze(mesh.vertices[mesh.faces[target_faces1]])
                target_v2 = mesh.vertices[mesh.faces[target_faces2]]
                vertice = mesh.vertices[v]
                target_normal1 = np.cross(target_v1[1] - target_v1[0], target_v1[2]-target_v1[1])
                target_normal1 = target_normal1 / np.linalg.norm(target_normal1)
                v_normal = np.cross(vertice[1] - vertice[0], vertice[2] - vertice[1])
                v_normal = v_normal / np.linalg.norm(v_normal)
                re1 = np.dot(faces_normal1[0], target_normal1)
                re2 = np.dot(faces_normal1[0], v_normal)
                # 判断法向量的朝向
                if re2 > 0:
                    return 0
                else:
                    return 1
                
                return 0

            ring_edges = edges
            hole_edges = vertices[torch.squeeze(ring_edges)]   # [n,2,3]每条边的每个顶点的坐标。
            hole_vector = hole_edges[:, 0, :] - hole_edges[:, 1, :]     # 所有边中，第一个顶点与第二个顶点之间的差，求得向量。
            hole_angle = []
            edges_num = len(ring_edges)
            # 计算相邻边的角度
            for i in range(edges_num):
                vector1 = hole_vector[i]
                vector2 = hole_vector[(i+1) % edges_num]
                angle_degrees = dot_product(vector1, vector2)
                hole_angle.append(angle_degrees)
                
            hole_angle = torch.tensor(hole_angle)
            new_faces = []
            normal_vector_flag = -1
            while len(hole_angle) - 3 != 0:
                # 找到角度最小的 下标索引
                min_index = torch.argmin(hole_angle)
                # 根据下标索引，找到顶点索引
                v1 = ring_edges[min_index][0].item()
                v2 = ring_edges[min_index][1].item()
                v3 = ring_edges[(min_index+1)%len(ring_edges)][1].item()
                # 根据下标索引，找到顶点坐标
                v_coor1 = hole_edges[min_index][0]
                v_coor2 = hole_edges[min_index][1]
                v_coor3 = hole_edges[(min_index+1)%len(hole_edges)][1]
                # 找到v1 v2 的这条边公用的面片，然后得到这个面片的标准法向量 n1
                # 在计算这个面片的 v1 v2 x 顺序的法向量n2，
                # 计算新生成的面片的法向量n3，顺序按照v1 v2 v3
                # 计算n1与n2方向是否相同，然后计算n1与n3方向是否相同，要求结果与一致
                # 添加新面片
                if normal_vector_flag == -1:
                    normal_vector_flag = verify_normal_vector(mesh,mask,[v1,v2,v3])
                if normal_vector_flag == 0:
                    new_faces.append([v1, v2, v3])
                else:
                    new_faces.append([v3, v2, v1])

                new_redges = torch.tensor([v1,v3]).unsqueeze(0)                  # 新边，点索引
                new_hedges = torch.stack([v_coor1, v_coor3]).unsqueeze(0)       # 新边，点坐标
                new_vector = v_coor1 - v_coor3     # 新向量
                new_angle1 = dot_product(hole_vector[(min_index-1)%len(hole_vector)], new_vector)   # 新角度1
                new_angle2 = dot_product(new_vector, hole_vector[(min_index+2)%len(hole_vector)])   # 新角度2
                new_vector = new_vector.unsqueeze(0)
                
                # 先进行删除并添加：
                if min_index == 0:
                    hole_angle = torch.cat((new_angle1, hole_angle[2:-1], new_angle2),dim=0)
                    ring_edges = torch.cat((ring_edges[:min_index], new_redges, ring_edges[min_index+2:]), dim=0)
                    hole_edges = torch.cat((hole_edges[:min_index], new_hedges, hole_edges[min_index+2:]), dim=0)
                    hole_vector = torch.cat((hole_vector[:min_index], new_vector, hole_vector[min_index+2:]), dim=0)
                elif min_index == len(ring_edges) - 1:
                    hole_angle = torch.cat((hole_angle[1:-2], new_angle1, new_angle2),dim=0)
                    ring_edges = torch.cat((ring_edges[1:-1], new_redges), dim=0)
                    hole_edges = torch.cat((hole_edges[1:-1], new_hedges), dim=0)
                    hole_vector = torch.cat((hole_vector[1:-1], new_vector), dim=0)
                else:
                    hole_angle = torch.cat((hole_angle[:min_index-1], new_angle1, new_angle2, hole_angle[min_index+2:]), dim=0) 
                    ring_edges = torch.cat((ring_edges[:min_index], new_redges, ring_edges[min_index+2:]), dim=0)
                    hole_edges = torch.cat((hole_edges[:min_index], new_hedges, hole_edges[min_index+2:]), dim=0)
                    hole_vector = torch.cat((hole_vector[:min_index], new_vector, hole_vector[min_index+2:]), dim=0)
                

            if normal_vector_flag == 0:
                new_faces.append([ring_edges[0][0], ring_edges[1][0], ring_edges[2][0]])
            else:
                new_faces.append([ring_edges[2][0], ring_edges[1][0], ring_edges[0][0]])
            new_faces = torch.tensor(new_faces)
            return new_faces

        print("start_DBScan")

        mesh = self.mesh.copy() # 对模型进行保存上色
        vertices = torch.tensor(mesh.vertices)
        faces = mesh.faces
        face_mid_points = np.sum(np.array(mesh.triangles), axis=1) # points : （n,3） n个三角形，每个三角形顶点坐标的和。
        face_mask_loss = (faces_loss > np.mean(faces_loss) * weight).astype(int) # 设置一个mask, 待处理的为1，否则为0
        # 用于存储面之间的邻接关系
        face_neighbor = []
        for i in range(len(face_mid_points)):
            face_neighbor.append([])
        for d in mesh.face_adjacency:
            face_neighbor[d[0]].append(d[1])
            face_neighbor[d[1]].append(d[0])

        diff_list = np.where(face_mask_loss == 1)[0].tolist()
        face_res = np.where(np.array(face_mask_loss) == 1)[0].tolist()
        face_res = np.array(face_res)
        res, cluster_num = getCluster(diff_list,face_neighbor)  # res:面片索引， cluster_num:初步选中的面片
        initial_color = [200, 200, 200, 0]  # 设置初始颜色，这里以白色为例
        mesh.visual.face_colors = initial_color
        mask = np.full(len(faces), True, dtype=bool)
        cluster = []
        new_faces = []
        num = 0
        for f in res:
            Dbscan_faces = faces[f]   # (n,3)
            Dbscan_edges = torch.tensor(trimesh.geometry.faces_to_edges(Dbscan_faces))    # (3n,3)
            Dbscan_edges = torch.sort(Dbscan_edges, dim=1).values
            # 统计每条边的出现次数
            
            unique_edges, counts = torch.unique(Dbscan_edges, dim=0, return_counts=True)
            single_occurrence_edges = unique_edges[counts == 1]
            ring_num, ring_edges = select_ring(single_occurrence_edges) # 环的数量，环的每条边（(n,2)，顶点索引）
            if ring_num == 2:
                # 补洞算法
                mask[f] = False
                # 将选中的面片输出并保存。
                for i in range(2):
                    ring_edges[i] = resort(ring_edges[i])
                    new_faces.append(fill_hole(ring_edges[i], vertices, mesh, mask))
                
                cluster.append(f)
                color = np.random.randint(0, 256, size=3).tolist() + [255]  # RGB + Alpha
                mesh.visual.face_colors[f] = color
                num += 1
        # mesh.export(self.out_dir + '/{}_DBScan1.ply'.format(self.dataname))
        print("select_num:{}, cluster_num:{}".format(num, cluster_num))
        print("spectral_clustering_over!!!!")

        if num > 0:
            new_faces = torch.cat(new_faces, dim=0)
            new_faces = new_faces.cpu().numpy()
            new_faces = np.vstack((faces[mask], new_faces))    # 以前的面片与新面片
            new_mesh = trimesh.Trimesh(mesh.vertices,new_faces, process=False)  # 生成新的网络模型
            new_mesh.remove_unreferenced_vertices()
            # new_mesh.export(self.out_dir + '/{}_new_mesh.ply'.format(self.dataname))
            self.mesh=new_mesh
        # res:[n,m] n：第n个簇，m：该簇中又m个面片
        for label in range(cluster_num):
            color = np.random.randint(0, 256, size=3).tolist() + [255]  # RGB + Alpha
            # print("res_face:", res[label])
            for f in res[label]:
                mesh.visual.face_colors[f] = color
        # print(cluster)

        # mesh.export(self.out_dir + '/{}_DBScan2.ply'.format(self.dataname))
        # print("spectral_clustering_over")

        # # 返回的cluster结果，如何转化成，res数组，(n,m,3)，n个聚类，每个聚类中m个顶点值。
        return num

    def ModifyTopo(self, cluster):
        # cluster:[n,m],n个簇，每个簇最多有m个面片
        from trimesh.constants import tol
        from trimesh import grouping
        from trimesh import util
        from sklearn.decomposition import PCA

        def p_normal(normals, iterations):

            plane_normal = torch.Tensor([1.0, 1.0, 1.0])    # 平面法向量
            # 求解平面法向量
            for _ in range(iterations):
                for n in normals:
                    plane_normal /= torch.norm(plane_normal)
                    # print("n:",n)
                    # print("plane:",plane_normal)
                    dot_product = torch.dot(n, plane_normal)
                    plane_normal -= dot_product * n / torch.norm(n)**2
                    # 正规化向量 
                    plane_normal /= torch.norm(plane_normal)
            
            return plane_normal

        def p_origin(vertice):
            plane_origin = torch.mean(vertice, dim=0)
            return plane_origin
        
        def modify_mesh(faces, vertices, plane_normal, plane_origin):

            def triangle_cases(signs_sorted):
                
                coded = torch.zeros(len(signs_sorted), dtype=torch.int8) + 14
                for i in range(3):
                    coded += signs_sorted[:, i] << 3 - i
                
                key = torch.zeros(29, dtype=torch.bool)
                key[[2,20]] = True    # [-1,-1,0],[0,1,1],[1,0,1],[1,1,0],20=24=26
                key[[6,16]] = True    # [-1,0,0],[0,0,1],[0,1,0],[1,0,0] ,16=18=22
                key[[8]] = True       # [-1,0,1],[-1,1,0] , 8=10
                key[14] = True              # [0,0,0]
                f_v_plane = key[coded.long()]      # 面片中，有顶点在平面上,f_v_p:[]

                key[:] = False
                key[[4,12]] = True         # [-1,-1,1],[-1,1,1]
                f_nov_plane = key[coded.long()]    # 面片中，没有顶点在平面上


                return f_v_plane, f_nov_plane

            plane_normal = torch.tensor(plane_normal, dtype=torch.float64)
            plane_origin = torch.tensor(plane_origin, dtype=torch.float64)
            
            # print(is_hole(vertices[faces]))
            dots = torch.matmul(vertices - plane_origin, plane_normal)

            signs = torch.zeros(len(vertices), dtype=torch.int8)

            signs[dots < -tol.merge] = -1
            signs[dots > tol.merge] = 1
            signs = signs[faces]     # 将[1,-1,1,1,-1,1]的六个顶点， 转变为：[[0,1,2],[2,3,4],[4,5,6]] -> [[1,-1,1],[1,1,-1],[1,-1,1]]       

            # 筛选出切割的部分
            # if select_ring(signs, faces):
            signs_sorted = torch.sort(signs, dim=1).values
            f_v_plane, f_nov_plane = triangle_cases(signs_sorted)  # 面片中，有顶点在平面上，和没有顶点在平面上（面片索引）
            
            signs_on_plane = signs[f_v_plane]           # 对应的 signs(有顶点)
            f_on_plane = faces[f_v_plane]               # 对应的 faces（有顶点）
            v_plane = f_on_plane[signs_on_plane == 0]   # 在平面上的顶点

            # signs_no_plane = signs[f_nov_plane]         # 对应的 signs（有顶点）
            f_no_plane = faces[f_nov_plane]             # 没有顶点在平面上的面片(顶点索引)

            return v_plane, f_no_plane, f_on_plane, len(f_no_plane)+len(f_on_plane)
            
            # else:
            #     return None,None,None,None,None,0

        def general_faces(vertices):
            faces = []
            for v in vertices:
                v = v.cpu().numpy()
                # print(v.shape)
                # v = torch.tensor(v)
                # indices = torch.arange(1, v.size(0) - 1)
                # face = torch.column_stack((v[0].repeat_interleave(indices.size(0)), v[indices], v[indices+1]))
                # faces.extend(face.cpu().numpy().tolist())
                print(v[0])
                for i in range(1, len(v) - 1):
                     faces.append([v[0], v[i], v[i+1]])
            return faces

        points = self.mesh.vertices.astype(np.float32) # 将xyz的顶点以numpy的形式保存在points中，并且不具有梯度信息，在cpu上运算
        normal_mesh = trimesh.Trimesh(vertices=points, faces=self.mesh.faces, process=False)
        normals = torch.FloatTensor(normal_mesh.face_normals).cuda()
        vertices = torch.FloatTensor(points).cuda() 
        normal_faces = torch.FloatTensor(normal_mesh.faces).cuda()

        v_plane = torch.Tensor()
        f_no_plane = torch.Tensor()
        f_on_plane = torch.Tensor()
        for idx in cluster:
            # print("idx:",idx)
            idx = torch.tensor(idx, dtype=torch.int64).cuda()
            idx_faces = torch.tensor(normal_faces[idx], dtype=torch.int64)
            idx_vertice = vertices[torch.unique(idx_faces)]    # idx对应的面片中，无重复的顶点坐标
            normal = normals[idx]                   # 获取每个面片的法向量。

            plane_normal = p_normal(normal,20)      # 获取每个簇的法向量
            if torch.all(plane_normal == torch.Tensor([0,0,0])):
                continue
            plane_origin = p_origin(idx_vertice)        # 获取每个簇的平面坐标点
            # 每个簇中，与平面相交的面片的顶点，分为在平面上的顶点，和不在平面上的顶点。
            v, fnovp, fonvp, num = modify_mesh(idx_faces, vertices, plane_normal, plane_origin)

            if num > 10:
                v_plane = torch.cat((v_plane,v), dim=0)           # 在平面上的顶点索引
                f_no_plane = torch.cat((f_no_plane,fnovp), dim=0)    # 不在平面上的面片，面片中是三个顶点索引
                f_on_plane = torch.cat((f_on_plane,fonvp), dim=0)    # 在平面上的面片，面片中是三个顶点索引

        face_mask = torch.ones(len(normal_faces), dtype=torch.bool)
        face_mas = torch.ones(len(normal_faces), dtype=torch.bool)


        # 获取b在a中的下标
        f_no_plane = torch.nonzero(torch.all(torch.isin(normal_faces, f_no_plane), dim=1)).squeeze(1)
        f_on_plane = torch.nonzero(torch.all(torch.isin(normal_faces, f_on_plane), dim=1)).squeeze(1)
        remove_face_len = len(f_no_plane) + len(f_on_plane)
        if remove_face_len == 0:
            return False

        print("remove_face:", remove_face_len)


        if f_no_plane.shape[0] != 0:
            face_mask[f_no_plane.long()] = False

        if f_on_plane.shape[0] != 0:
            face_mask[f_on_plane.long()] = False

        new_faces = normal_faces[face_mask].cpu().numpy()
        new_color = [255, 0, 0, 255]
        mesh = self.mesh
        mesh.visual.vertex_colors = [0,0,255,255]
        mesh.visual.face_colors[face_mas.cpu().numpy()] = new_color
        # mesh.export(self.out_dir + '/{}_selected_face.ply'.format(self.dataname))
        # general_face = general_faces(v_up_plane)
        # # print(general_face)
        # general_face.extend(general_faces(v_dw_plane))
        # general_face = np.array(general_face)
        # new_faces = np.vstack((new_faces, general_face))

        # self.mesh = trimesh.Trimesh(vertices=points, faces=new_faces)
        face_mask = face_mask.cpu().numpy()
        self.mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()

        # if v_plane.shape[0] != 0:
        #     vertices_mask = torch.zeros(len(self.mesh.vertices), dtype=bool)
        #     vertices_mask[new_faces] = True
        #     vertices_mask = vertices_mask.cpu().numpy()
        #     self.mesh.update_vertices(vertices_mask)

        return True

    def ModifyTopo_test(self, mask):

        # print("after_f:", len(self.mesh.faces))
        # print("after_v:", len(self.mesh.vertices))

        self.mesh.update_faces(mask)
        self.mesh.remove_unreferenced_vertices()

        # print("later_f:", len(self.mesh.faces))
        # print("later_v:", len(self.mesh.vertices))


        return True






def main(args):
    import time
    # 开始
    start_time = time.time()
    # 将命令行参数 args.dataname 赋给 args.dir_name，
    args.dir_name = args.dataname
    # 指定的GPU
    torch.cuda.set_device(args.gpu)
    # 配置文件的路径
    conf_path = args.conf
    # 打开配置文件
    f = open(conf_path)
    # 将文件内容存储在 conf_text变量中
    conf_text = f.read()
    f.close()
    device = torch.device('cuda')   # 用于确定模型和数据在GPU上运行

    conf = ConfigFactory.parse_string(conf_text)
    udf_network = UDFNetwork(**conf['model.udf_network']).to(device)    # 自定义了一个神经网络
    checkpoint_name = conf.get_string('evaluate.load_ckpt') # 取出evaluate.load_ckpt的参数值，用于加载与预训练模型的检查点文件的名称？
    base_exp_dir = conf['general.base_exp_dir'] + args.dataname # 基础目录 + 数据文件
    checkpoint = torch.load(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name),
                            map_location=device)    # 加载预训练模型的检查点文件，并将其存储在checkpoint变量中。
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
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    evaluator = Evaluator(lambda pts: udf_network.udf(pts),conf['general.base_exp_dir'], args.dataname,
                          conf.get_int('evaluate.resolution'), conf.get_float('evaluate.threshold'),
                          max_iter=conf.get_int("evaluate.max_iter"),normal_step=conf.get_int("evaluate.normal_step"),laplacian_weight=conf.get_int("evaluate.laplacian_weight"),bound_min=object_bbox_min,bound_max=object_bbox_max,
                          is_cut=True, region_rate=20,
                          max_batch=conf.get_int("evaluate.max_batch"),learning_rate=conf.get_float("evaluate.learning_rate"), warm_up_end=conf.get_int("evaluate.warm_up_end"), use_exist_mesh=False,
                          save_iter=conf.get_int("evaluate.save_iter"), report_freq=conf.get_int("evaluate.report_freq"),export_grad_field = conf.get_int("evaluate.export_grad_field"))

    with torch.autograd.detect_anomaly():
        evaluator.evaluate()
    # 结束
    end_time = time.time()
    print("time cost: {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()  # 用于解析命令行参数。
    # 配置文件路径
    parser.add_argument('--conf', type=str, default='./confs/base.conf')    # 提取命令行中的值， --xxx表示可选 
    # MC的分辨率，默认256
    parser.add_argument('--mcube_resolution', type=int, default=256)
    # gpu设备的选择， 默认0号
    parser.add_argument('--gpu', type=int, default=0)
    # 指定数据的名称。？需要看一下文件结构，才能更好的理解。
    parser.add_argument('--dataname', type=str, default='demo')
    # 解析命令行参数，结果保存到args中
    args = parser.parse_args()
    main(args)
