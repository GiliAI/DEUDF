import torch
import numpy as np
import igl

from helper import get_path_for_data
from data.dataset import CDataset
from scipy.spatial import KDTree
from network.siren import gradient
import pymeshlab
import trimesh
from evaluator.optimize_A import getMeshUDF
from DCUDF.dcudf.VectorAdam import VectorAdam
import math
from task.chamfer import NP_Cosine,get_color


def load_point_cloud(path, num_points,hasgtnormal=True):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    coords = mesh.vertex_matrix().astype('float32')
    gt_normals = mesh.vertex_normal_matrix().astype('float32')
    if num_points < coords.shape[0]:
        idx = np.random.permutation(coords.shape[0])[:num_points]
        coords = np.ascontiguousarray(coords[idx])
        gt_normals = np.ascontiguousarray(gt_normals[idx])


    mesh = pymeshlab.Mesh(vertex_matrix=coords)
    ms.add_mesh(mesh,"small_mesh")
    if hasgtnormal:
        normals = gt_normals
    else:
        ms.compute_normal_for_point_clouds()
        normals = ms.current_mesh().vertex_normal_matrix().astype('float32')

    return coords, normals, gt_normals

def addGaussianNoise(coords, normals, propotion=1, bound=0.01):
    result_coords = coords
    result_normals = normals

    noise_size = int(coords.shape[0]*propotion)
    noise = torch.randn((noise_size,3))*bound
    rand_id = np.random.choice(coords.shape[0],size=noise_size)

    result_coords[rand_id] += noise

    return result_coords, result_normals



class ScPointcloud(CDataset):
    """
    Attributes:
        num_points: load total of N points from a dense point cloud
        batch_size: total number of on-and-off-surface query points for training SDF
        pointcloud_size: number of on-surface points for encoding
        keep_aspect_ratio: normalization
        factor_off_surface: ratio of off-surface points
    """
    def __init__(self, config):
        self.path : str = None
        self.pointcloud_path : str = None
        self.num_points : int = 1000000
        self.train_epoch_points : int = 1000000
        self.batch_size : int = 100000
        self.pointcloud_size: int = 3000
        self.keep_aspect_ratio : bool = True
        self.factor_off_surface : float = 0.5
        self.bbox_size : float = 2.0
        self.padding : float = 0.1

        self._has_init : bool = False
        self._coords  : np.ndarray = None
        self._normals : np.ndarray = None

        self.enhance_mask : np.ndarray = None
        self.offset : float = 0.002

        self.addNoise : bool = False
        self.hasgtnormal : bool = True
        

        super().__init__(config)
        if self.pointcloud_path is None:
            self.pointcloud_path = self.path

    def _init(self):
        if(self._has_init):
            return
        path = get_path_for_data(self, self.path)
        self.runner.py_logger.info(f"Loading pointcloud from {path}\n")


        self._coords, self._normals, gt_normals = load_point_cloud(path, self.num_points,hasgtnormal = self.hasgtnormal)

        self._coords = torch.from_numpy(self._coords)
        self._normals = torch.from_numpy(self._normals)
        self.normalize(self._coords, self._normals)

        self._zerobasecoords = torch.zeros_like(self._coords)
        self._zerodiscoords = torch.zeros_like(self._coords)

        self.coordsTree = KDTree(self._coords.clone().detach().numpy())

        self.enhance_mask = torch.ones((self._coords.shape[0],1))

        if self.addNoise:
            self._coords, self._normals = addGaussianNoise(self._coords, self._normals)

        if self.pointcloud_path != self.path:
            path = get_path_for_data(self, self.pointcloud_path)
            self.runner.py_logger.info(f"Loading pointcloud points from {path}\n")
            pcl_coords, pcl_normals = load_point_cloud(path,hasgtnormal = self.hasgtnormal)
            pcl_coords = torch.from_numpy(pcl_coords)
            pcl_normals = torch.from_numpy(pcl_normals)
            self.normalize(pcl_coords, pcl_normals)
            self.pcl = torch.cat([pcl_coords, pcl_normals], dim=-1)
        else:
            self.pcl = torch.cat([self._coords, self._normals], dim=-1)


        nd = NP_Cosine(gt_normals,self._normals.numpy())
        nd = (1-np.abs((1-nd*2)))/2
        self.runner.logger.log_mesh("pointclouddata", 
                                    vertices = self._coords[None,...], 
                                    faces = None, 
                                    colors = get_color(nd,"cool",nd.min(),nd.max()).reshape(1,-1,3),
                                    vertex_normals=self._normals[None,...])


        self._has_init = True

    def __len__(self):
        self._init()
        # return (self._coords.shape[0] // int(self.batch_size*(1-self.factor_off_surface))) + 1
        return (self.train_epoch_points // int(self.batch_size*(1-self.factor_off_surface))) + 1

    def __getitem__(self, idx):
        self._init()

        point_cloud_size = self._coords.shape[0]

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        total_samples = self.batch_size
        on_surface_samples = self.batch_size - off_surface_samples


        
        # Random coords
        rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
                                        size=on_surface_samples))

        on_surface_coords = self._coords[rand_idcs, :]
        on_surface_normals = self._normals[rand_idcs, :]

        zero_base_coords = self._zerobasecoords[rand_idcs, :]
        zero_dis_coords = self._zerodiscoords[rand_idcs, :]

        on_surface_mask = self.enhance_mask[rand_idcs, :]
        on_surface_nearest = torch.cat([on_surface_coords,on_surface_normals],dim=-1)

        offset = (torch.rand((on_surface_normals.shape[0],1)))*self.offset
        on_suface_coords_l = on_surface_coords -  offset.repeat(1,3)*on_surface_normals
        on_suface_coords_r = on_surface_coords +  offset.repeat(1,3)*on_surface_normals
        
        
        sdf = torch.zeros((total_samples, 1))  # on-surface = 0

        pointcloud = torch.zeros((self.pointcloud_size, 3))

        if self.pointcloud_size > 0:
            # sample randomly from point cloud
            rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
                                     size=self.pointcloud_size))
            pointcloud = self.pcl[rand_idcs, :]


        if(off_surface_samples > 0):
            off_surface_coords = torch.rand(off_surface_samples, 3) - 0.5
            off_surface_normals = torch.ones((off_surface_samples, 3)) * -1

            off_surface_masks = torch.ones((off_surface_samples, 1))

            off_surface_coords *= self.bbox_size


            sdf[on_surface_samples:, :] = -1  # off-surface = -1

            coords = torch.cat((on_surface_coords, off_surface_coords),
                                    dim=0)
            normals = torch.cat((on_surface_normals, off_surface_normals),
                                    dim=0)

            offset = torch.cat((offset,torch.zeros(off_surface_coords.shape[0],1)),dim=0)
            coords_l = torch.cat((on_suface_coords_l, off_surface_coords),
                                    dim=0)
            coords_r = torch.cat((on_suface_coords_r, off_surface_normals),
                                    dim=0)
            
            zero_base_coords = torch.cat((zero_base_coords, off_surface_coords),
                                    dim=0)
            zero_dis_coords = torch.cat((zero_dis_coords, off_surface_coords),
                                    dim=0)

            
            masks = torch.cat((on_surface_mask,off_surface_masks),dim=0)


        else:
            coords = on_surface_coords
            normals = on_surface_normals

            masks = on_surface_mask

        return {
                "coords" : coords,
                "normal_out" : normals,
                "sdf_out" : sdf,
                "pointcloud": pointcloud,
                "masks" : masks,
                "zero_base_coords":zero_base_coords,
                "zero_dis_coords":zero_dis_coords,
                "coords_l": coords_l,
                "coords_r": coords_r,
                "offset": offset

                }
    
   
    def update_learning_rate(self,iter_step,max_iter,init_lr,optimizer,warn_up=25):
        # warn_up = 25
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        if iter_step>=200:
            lr *= 0.1
        for g in optimizer.param_groups:
            g['lr'] = lr

