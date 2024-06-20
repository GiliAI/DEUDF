# -*- coding: utf-8 -*-
import json
import time
import torch
from run_2 import Runner
import argparse

import warnings
warnings.filterwarnings('ignore')


        
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
