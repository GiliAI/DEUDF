# -*- coding: utf-8 -*-

import time
import torch

import argparse


import warnings
warnings.filterwarnings('ignore')
import json
from evaluate_custom import main
        
if __name__ == '__main__':


    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_resolution', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_file', type=str, required=True)
    args = parser.parse_args()



    torch.cuda.set_device('cuda:'+str(args.gpu))

    cost = []
    with open(args.batch_file, "r") as f:
        paths = json.load(f)
    for dataname in paths:
        start_time = time.time()
        args.dataname = dataname
        args.dir_name = dataname
        print(dataname)
        main(args)
    with open("time.txt", "w") as f:
        for t in cost:
            f.write(str(t))
            f.write("\n")
