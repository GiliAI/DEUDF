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


class DrawPlane(Evaluator):

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
    
        # xlist = np.linspace(-0.5, 0.5, self.resolution) * bbox_size
        # ylist = np.linspace(-0.5, 0.5, self.resolution) * bbox_size
        # X, Y = np.meshgrid(xlist, ylist)
        # Z = np.ones_like(X)*self.offset

        # if(self.axis == 0):
        #     coords = np.stack([Z, Y, X], 2)
        # elif (self.axis == 1):
        #     coords = np.stack([X, Z, Y], 2)
        # else:
        #     coords = np.stack([X, Y, Z], 2)
        ylist = np.linspace(-0.05, 0.05, self.resolution)    
        coords = np.zeros([self.resolution,3])
        coords[:,1] = ylist

        coords = coords.reshape([-1,3])
        # num_batches = int((self.batch_size -1 + self.resolution**2)/self.batch_size)
        # results = {}
        l_coords = torch.Tensor(coords).cuda()
        eval_results =  self.evaluate_network(l_coords.unsqueeze(0), fea=self.encode_network(l_coords.unsqueeze(0)), **data)


        # 一元一次函数图像
        x = ylist
        y = eval_results["sdf"].cpu().detach().numpy().flatten()


        # plt.title("plane_")


        # fig = plt.figure()
        # plt.plot(x, y)
        # # plt.plot(x, np.abs(x))
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20) 
        # # plt.xlabel('input_y_axis')
        # # plt.ylabel('output_udf')
        # # plt.tick_params(labelsize=20)
        # plt.xlim(-0.05, 0.05)
        # plt.ylim(-0.02, 0.08)

        # name = f"{self.name}"
        # self.runner.logger.log_figure(name, fig)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_proj_type('persp')        
        # yline = coords
        yb = ylist[(ylist>=0)&(ylist<0.038)]
        yf = ylist[(ylist<0)&(ylist>=-0.038)]
        yb2 = ylist[ylist>=0.038]
        yf2 = ylist[ylist<-0.038]

        xlist = np.linspace(-0.5, 0.5, self.resolution) * 2
        zlist = np.linspace(-0.5, 0.5, self.resolution) * 2
        X, Z = np.meshgrid(xlist, zlist)
        ax.plot_surface(X=X,Y=0*X,Z=Z,color="b",alpha=0.3)
        ax.plot3D(xs=np.zeros_like(yb),ys=yb,zs=np.zeros_like(yb),color="r",alpha=0.3)
        ax.plot3D(xs=np.zeros_like(yf),ys=yf,zs=np.zeros_like(yf),color="r",alpha=1.0)
        ax.plot3D(xs=np.zeros_like(yb2),ys=yb2,zs=np.zeros_like(yb2),color="r",alpha=1.0)
        ax.plot3D(xs=np.zeros_like(yf2),ys=yf2,zs=np.zeros_like(yf2),color="r",alpha=0.8)

        ax.tick_params(labelsize=15)
        ax.tick_params(axis='y',labelsize=12)
        ax.set_xticks([-0.8,-0.4,0,0.4,0.8])
        ax.set_zticks([-0.8,-0.4,0,0.4,0.8])
        ax.set_yticks([-0.04,-0.02,0,0.02,0.04])
        # ax.tick_params(axis='y',labelsize=10)

        # ax.set_xlabel('x',size=15)
        # ax.set_ylabel('y',size=15)
        # ax.set_zlabel('z',size=15)
        ax.view_init(azim=-35)

        name = f"{self.name}"
        self.runner.logger.log_figure(name, fig)





        # plt.show()

        # for attr in self.attributes :
        #     result = results[attr]
        #     name = f"{self.name}_{attr}"
        #     sdf = result.reshape(X.shape)
        #     fig = plt.figure()


        #     plt.contour(X, Y, sdf, levels=bounds[attr])
        #     plt.contourf(X, Y, sdf, levels=bounds[attr], norm=norms[attr], cmap=cmap)
        #     if 'gt' in self.attributes:
        #         plt.contour(X,Y, results['gt'].reshape([self.resolution, self.resolution]),[0], colors='yellow', linestyles='dotted')
        #     plt.colorbar(extend="max")

            
        #     self.runner.logger.log_figure(name, fig)

    def epoch_hook(self, epoch : int, data:Dict=None):
        if(epoch % self.frequency == 0):
            self.runner.py_logger.info(f"Generating plane plot {self.name} with resolution {self.resolution}")
            self.evaluate_level(data)