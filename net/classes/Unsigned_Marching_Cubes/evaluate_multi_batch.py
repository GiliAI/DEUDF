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

from models.dataset import Dataset
from models.fields import CAPUDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile

from tools.logger import get_logger, get_root_logger, print_log
from tools.utils import remove_far, remove_outlier
from tools.surface_extraction import as_mesh, surface_extraction, threshold_MC

import point_cloud_utils as pcu

import warnings
warnings.filterwarnings('ignore')
import json
from evaluate_multi import Evaluator
        
if __name__ == '__main__':
    import time


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
        evaluator = Evaluator(args, args.conf,centers=[0,0,0],total_size=1)

        evaluator.evaluate()
        end_time = time.time()
        cost.append( end_time - start_time)
        print(cost[-1])
    with open("time.txt", "w") as f:
        for t in cost:
            f.write(str(t))
            f.write("\n")
