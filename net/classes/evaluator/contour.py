from typing import Dict
from evaluator.evaluator import Evaluator
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


cmap = plt.get_cmap('PuOr')
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize and forcing 0 to be part of the colorbar!
# sdf_bounds = np.arange(-1, 1, 0.1)
# addition
base_bounds = np.arange(0, 1, 0.05)
base_norm = BoundaryNorm(base_bounds, cmap.N)

residual_bounds = np.arange(-0.1, 0.1, 0.01)
residual_norm = BoundaryNorm(residual_bounds, cmap.N)

sigma_bounds = np.arange(0.01, 0.3, 0.01)
sigma_norm = BoundaryNorm(sigma_bounds, cmap.N)

mid_bounds = np.arange(-1, 1, 0.1)
mid_norm = BoundaryNorm(mid_bounds, cmap.N)

bounds = {"sdf":base_bounds,"residual":residual_bounds,"base":base_bounds,"sigmas":sigma_bounds,"mid":mid_bounds}
norms = {"sdf":base_norm,"residual":residual_norm,"base":base_norm,"sigmas":sigma_norm,"mid":mid_norm}



class Contour(Evaluator):

    def __init__(self, config):
        self.attributes = ["sdf"]
        self.frequency =  10
        self.resolution = 200
        self.axis = 0
        self.offset = 0
        self.batch_size = 125000
        self.bbox_size :float = 2.0
        super().__init__(config)

    def evaluate_level(self, data):
        data.pop('coords', None)
        data.pop('featurepoints',None)
        # fea = self.encode_network(data)

        torch.cuda.empty_cache()
        try:
            bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        except:
            bbox_size = self.bbox_size

        
        # self.runner.network.iniFeatures()
        try:
            self.runner.network.iniFeatures()
        except Exception as e:
            print(e)
            print("init Feature failed,if no features,ignore this message")
    
        xlist = np.linspace(-0.5, 0.5, self.resolution) * bbox_size
        ylist = np.linspace(-0.5, 0.5, self.resolution) * bbox_size
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.ones_like(X)*self.offset
        if(self.axis == 0):
            coords = np.stack([Z, Y, X], 2)
        elif (self.axis == 1):
            coords = np.stack([X, Z, Y], 2)
        else:
            coords = np.stack([X, Y, Z], 2)

        coords = coords.reshape([-1,3])
        num_batches = int((self.batch_size -1 + self.resolution**2)/self.batch_size)
        results = {}
        for i in range(num_batches):
            start = i*self.batch_size
            end = min(self.resolution**2-1,start+self.batch_size)
            l_coords = torch.Tensor(coords[start:end,:]).cuda()
            eval_results =  self.evaluate_network(l_coords.unsqueeze(0), fea=self.encode_network(l_coords.unsqueeze(0)), **data)

            for attr in self.attributes:
                if i==0:
                    results[attr] = np.zeros([self.resolution**2])
                result = eval_results[attr]
                results[attr][start:end] = result.cpu().detach().numpy().flatten()
                del result
            del eval_results
            del l_coords

        for attr in self.attributes :
            result = results[attr]
            name = f"{self.name}_{attr}"
            sdf = result.reshape(X.shape)
            fig = plt.figure()

            # fig = plt.figure(figsize=(fig.bbox_inches.width*2,fig.bbox_inches.height))
            # plt.subplot(121)

            # plt.contour(X, Y, sdf, levels=udf_bounds)
            # plt.contourf(X, Y, sdf, levels=udf_bounds, norm=norm, cmap=cmap)
            plt.contour(X, Y, sdf, levels=bounds[attr])
            plt.contourf(X, Y, sdf, levels=bounds[attr], norm=norms[attr], cmap=cmap)
            if 'gt' in self.attributes:
                plt.contour(X,Y, results['gt'].reshape([self.resolution, self.resolution]),[0], colors='yellow', linestyles='dotted')
            plt.colorbar(extend="max")

            # plt.subplot(122)
            # sdf_notzero = sdf[sdf>0]
            # sdf_min = 0
            # if sdf_notzero.size > 0:
            #     sdf_min = np.min(sdf_notzero)
            # zero_level_coords = np.where((sdf <=sdf_min*1.2) & (sdf >0))
            # if(self.axis == 0):
            #     plt.scatter(xlist[zero_level_coords[1]],ylist[zero_level_coords[0]],s=0.05,c = 'r',marker="o")
            # elif (self.axis == 1):
            #     plt.scatter(ylist[zero_level_coords[1]],xlist[zero_level_coords[0]],s=0.05,c = 'r',marker="o")
            # else:
            #     plt.scatter(ylist[zero_level_coords[1]],xlist[zero_level_coords[0]],s=0.05,c = 'r',marker="o")
            
            self.runner.logger.log_figure(name, fig)

    def epoch_hook(self, epoch : int, data:Dict=None):
        if(epoch % self.frequency == 0):
            self.runner.py_logger.info(f"Generating contour plot {self.name} with resolution {self.resolution}")
            self.evaluate_level(data)