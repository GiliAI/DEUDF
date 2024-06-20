# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import Dataset
from models.fields_origin import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from tools.logger import get_logger, get_root_logger, print_log
from tools.utils import remove_far, remove_outlier
from tools.surface_extraction import as_mesh, surface_extraction
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
from scipy.spatial import cKDTree
import point_cloud_utils as pcu
import csv
import warnings
import json

warnings.filterwarnings('ignore')

def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
    print("1111")
    s1 = time.time()
    N = 32
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

                grad = grad_func(pts).reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                g[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = grad
    s2 = time.time()
    print("2222")
    print(s2-s1)
    # with open("time_extract.txt", "a") as f:
    #     f.write(str(s2-s1))
    #     f.write("\n")
    return u, g

def extract_geometry(bound_min, bound_max, resolution, npz_path, out_dir, iter_step, dataname, logger):

    print('Extracting mesh with resolution: {}'.format(resolution))
    data = np.load(npz_path)
    u = data["df"]
    g = data["grad"]

    b_max = data["bound_max"]
    b_min = data["bound_min"]
    mesh = surface_extraction(u, g, out_dir, iter_step, b_max, b_min, resolution)

    return mesh

class Runner:
    def __init__(self, args, conf_path):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        # self.dataset = Dataset(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters

        # Backup codes and configs for debug
        self.file_backup()

    def evaluate(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.extract_mesh(resolution=args.mcube_resolution, threshold=0.0, point_gt=None, iter_step=self.iter_step)


    def extract_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):

        bound_min = torch.tensor((-1,-1,-1), dtype=torch.float32)
        bound_max = torch.tensor((1,1,1), dtype=torch.float32)
        out_dir = os.path.join(self.base_exp_dir, 'mesh')
        os.makedirs(out_dir, exist_ok=True)

        mesh = extract_geometry(bound_min, bound_max, resolution=256, npz_path=os.path.join(self.base_exp_dir,"field_0.0025.npz"), \
                out_dir=out_dir, iter_step=iter_step, dataname=self.dataname, logger=logger)
        mesh.process()
        mesh.export(out_dir+'/'+str(iter_step)+'_mesh.ply')




    def update_learning_rate(self, iter_step):

        warn_up = self.warm_up_end
        max_iter = self.step2_maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr

        for g in self.optimizer.param_groups:
            g['lr'] = lr
            
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.udf_network.load_state_dict(checkpoint['udf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'udf_network_fine': self.udf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/ndf.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_file', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    cost = []
    with open(args.batch_file, "r") as f:
        paths = json.load(f)
    for dataname in paths:
        s = time.time()
        args.dataname = dataname
        args.dir = dataname
        print(dataname)
        runner = Runner(args, args.conf)

        runner.evaluate()
        e = time.time()
        cost.append(e-s)
        print(cost[-1])
    with open("time_capudf.txt", "w") as f:
        for t in cost:
            f.write(str(t))
            f.write("\n")

