# -*- coding: utf-8 -*-
import json
import time
import torch
from tqdm import tqdm
from models.dataset_2_multi import Dataset
from models.fields import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh

from evaluate_multi import Evaluator
from tools.logger import get_logger, get_root_logger, print_log
from tools.utils import remove_far, remove_outlier
from tools.surface_extraction import threshold_MC
import warnings
warnings.filterwarnings('ignore')

def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
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

    return u, g

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, grad_func):

    u, g = extract_fields(bound_min, bound_max, resolution, query_func, grad_func)
    mesh = threshold_MC(u, threshold, resolution)

    return mesh

class Runner:
    def __init__(self, args, conf_path):

        self.device = torch.device('cuda')

        # Configuration
        self.args = args
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'] + args.dataname)
        os.makedirs(self.base_exp_dir, exist_ok=True)

        
        self.dataset = Dataset(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.max_iter = self.conf.get_int('train.max_iter')
        self.max_iter_2 = self.conf.get_int("train.max_iter_2")
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.batch_size_step2 = self.conf.get_int('train.batch_size_step2')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)

        #Evaluating parameters
        self.resolution = self.conf.get_int("train.resolution")
        self.threshold = self.conf.get_float("train.threshold")

        self.loss_l1 = torch.nn.MSELoss()

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.model_list = []
        self.writer = None

        # Networks
        self.udf_network = CAPUDFNetwork(**self.conf['model.udf_network']).to(self.device)
        if self.conf.get_string('train.load_ckpt') != 'none':
            self.udf_network.load_state_dict(torch.load(self.conf.get_string('train.load_ckpt'), map_location=self.device)["udf_network_fine"])

        self.optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        self.file_backup()

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size
        if os.path.exists(os.path.join(self.base_exp_dir, "mesh", "mesh.ply")):
            print("Skip {}".format(self.args.dataname))
            return

        for iter_i in tqdm(range(self.iter_step,self.max_iter_2)):
            self.update_learning_rate(self.iter_step)
            samples, ndf = self.dataset.get_train_data(batch_size)
            # samples.requires_grad = True
            # gradients_sample = self.udf_network.gradient(samples).squeeze()
            # grad_loss = self.loss_l1(gradients_sample, grad)
            udf_sample = self.udf_network.udf(samples)                      # 5000x1
            loss_cd = self.loss_l1(udf_sample, ndf)

            loss = loss_cd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l2 = {} lr={}'.format(self.iter_step, loss_cd, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_mesh_freq ==0:
                print(f"Extracting mesh using sample MC with {self.resolution} in {self.threshold}")
                self.simple_extract_mesh(iter_i)
            if self.iter_step == self.max_iter_2:
                evaluate =  Evaluator(self.args, self.conf_path, bounds=self.dataset.bounds, udf_network=self.udf_network, centers=self.dataset.centers,
                                      total_size=self.dataset.total_size)
                mesh = evaluate.evaluate()
                batch_size = self.batch_size_step2
                self.learning_rate = 0.1*self.learning_rate
                # self.dataset.gen_new_data_from_mesh(mesh)

    def simple_extract_mesh(self, iter_step=0, logger=None):

        bound_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
        bound_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        out_dir = os.path.join(self.base_exp_dir, 'mesh')
        os.makedirs(out_dir, exist_ok=True)

        mesh = extract_geometry(self.dataset.bounds[0], self.dataset.bounds[1], resolution=self.resolution, threshold=self.threshold,
                query_func=lambda pts: self.udf_network.udf(pts), grad_func=lambda pts: self.udf_network.gradient(pts))
        if mesh is not None:
            mesh.export(out_dir+'/'+str(iter_step)+'_mesh.obj')

    def update_learning_rate(self, iter_step):

        warn_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 1
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
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_resolution', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_file', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.set_device('cuda:'+str(args.gpu))
    with open(args.batch_file, "r") as f:
        paths = json.load(f)
    for dataname in paths:
        args.dataname = dataname
        print(dataname)
        runner = Runner(args, args.conf)

        runner.train()
