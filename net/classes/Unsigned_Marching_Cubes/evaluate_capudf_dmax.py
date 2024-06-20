# -*- coding: utf-8 -*-

import time
import torch


from models.fields_origin import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from evaluate_custom import Evaluator
import trimesh
import warnings
import numpy as np
warnings.filterwarnings('ignore')



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
    udf_network = CAPUDFNetwork(**conf['model.udf_network']).to(device)
    checkpoint_name = conf.get_string('evaluate.load_ckpt')
    base_exp_dir = conf['general.base_exp_dir'] + args.dataname
    checkpoint = torch.load(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name),
                            map_location=device)
    print(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name))
    udf_network.load_state_dict(checkpoint['udf_network_fine'])
    mesh_path = os.path.join(conf['dataset'].data_dir,"input","{}.ply".format(args.dataname))
    mesh = trimesh.load_mesh(mesh_path)
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)
    v = torch.from_numpy(np.asarray(mesh.sample(100000))).float().cuda(0)
    udf = udf_network.udf(v).detach().cpu().numpy().flatten()
    udf = np.sort(udf)
    print(udf [-1000:].mean())
    print(udf.max())
    print(udf.min())
    print(udf.mean())


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='demo')
    args = parser.parse_args()
    main(args)