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
    if isinstance(mesh, trimesh.PointCloud):
        pointcloud = mesh.vertices
    else:
        pointcloud = mesh.sample(1000000)
    shape_scale = np.max(
        [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
         np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
    shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                    (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                    (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale
    object_bbox_min = np.array(
        [np.min(pointcloud[:, 0]), np.min(pointcloud[:, 1]), np.min(pointcloud[:, 2])]) - 0.05
    object_bbox_max = np.array(
        [np.max(pointcloud[:, 0]), np.max(pointcloud[:, 1]), np.max(pointcloud[:, 2])]) + 0.05

    evaluator = Evaluator(args.dataname, lambda pts: udf_network.udf(pts), conf_path=args.conf,
                          bound_min=object_bbox_min,bound_max=object_bbox_max)

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