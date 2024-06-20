# -*- coding: utf-8 -*-


import torch

from models.fields_box import CAPUDFNetwork
from evaluate_custom import Evaluator
import argparse
from pyhocon import ConfigFactory

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import time

    start_time = time.time()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')
    conf_path = args.conf
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)
    udf_network = CAPUDFNetwork(**conf['model.udf_network']).to(device)
    query_func = lambda pts: udf_network.udf(pts)
    evaluator = Evaluator(args.dataname, query_func, conf_path=args.conf)

    evaluator.evaluate()
    end_time = time.time()
    print("time cost: {:.2f}s".format(end_time - start_time))
