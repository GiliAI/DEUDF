import torch
import numpy as np
import igl

from helper import get_path_for_data
from data.dataset import CDataset
from network.network import Network
from pykdtree.kdtree import KDTree
from typing import Dict
import pymeshlab
from network.siren import gradient
import trimesh

def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


def load_point_cloud(path, num_points):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)

    ms.compute_matrix_from_rotation(rotaxis='Z axis',angle=90)

    mesh = ms.current_mesh()

    coords = mesh.vertex_matrix().astype('float32')
    normals = mesh.vertex_normal_matrix().astype('float32')


    if num_points < coords.shape[0]:
        idx = np.random.permutation(coords.shape[0])[:num_points]
        coords = np.ascontiguousarray(coords[idx])
        normals = np.ascontiguousarray(normals[idx])

    # return coords, normals, face_coords, faces
    # coords = coords * -1
    # normals = normals * -1

    return coords, normals

def get_faces(_coords,_normals):
    m = pymeshlab.Mesh(vertex_matrix=_coords, v_normals_matrix=_normals)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "temp")
    ms.generate_surface_reconstruction_screened_poisson()
    mesh = ms.current_mesh()

    return mesh.vertex_matrix(),mesh.face_matrix()

def cal_off_point_cloud(query_coords,has_feature_Ids,_coords,_faces):
    # has_coord_Ids=np.where(has_feature!=0)[0]
    temp_query_coords = query_coords[has_feature_Ids]

    # test = igl.signed_distance(np.array([[-0.101816,-0.00348149,0.0665723]]),_coords, _faces)

    udf = np.ones([query_coords.shape[0],],dtype= float)*-1
    udf[has_feature_Ids] = np.abs(igl.signed_distance(temp_query_coords,_coords, _faces)[0])

    # mesh = trimesh.Trimesh(_coords.reshape([-1,3]),_faces.reshape([-1,3]), process=False)
    # mesh.export("poisson_mesh.ply")

    return udf

class LatentPointcloud(CDataset):
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
        self.batch_size : int = 100000
        self.pointcloud_size: int = 3000
        self.keep_aspect_ratio : bool = True
        self.factor_off_surface : float = 0.5
        self.bbox_size : float = 2.0
        self.padding : float = 0.1

        self._has_init : bool = False
        self._coords  : np.ndarray = None
        self._normals : np.ndarray = None


        # addition
        self._faces : np.ndarray = None
        self._face_coords : np.ndarray = None
        self._gridPoints : np.ndarray = None
        self._gridIDs : np.ndarray = None
        self.resolution = 2
        # self._offcoords : np.ndarray = None
        self._gridkdtree = None
        # each grid points
        self._gridFeaturePoints : np.ndarray = None
        # each grid's voxels' feature coords
        self._gridVoxelsFeaturePoints : np.ndarray = None
        # whether a grid has voxel features
        self._gridHasFeature : np.ndarray = None
        self._gridVoxelSize : np.ndarray = None

        self.num_off_points : int = 1000000
        self._off_coords : np.ndarray = None
        self._off_udf  : np.ndarray = None
        # self._offkdtree = None
        # self._onkdtree = None

        self.isEncoding : bool = True
        self.featureSize : int = 8
        self.voxelNum : int = 8

        super().__init__(config)
        if self.pointcloud_path is None:
            self.pointcloud_path = self.path

    def _init(self):
        if(self._has_init):
            return
        path = get_path_for_data(self, self.path)
        self.runner.py_logger.info(f"Loading pointcloud from {path}\n")

        self._coords, self._normals = load_point_cloud(path, self.num_points)

        if self.num_points != self._coords.shape[0]:
            self.num_points = self._coords.shape[0]

        self._coords = torch.from_numpy(self._coords)
        self._normals = torch.from_numpy(self._normals)
        self.normalize(self._coords, self._normals)
        if self.pointcloud_path != self.path:
            path = get_path_for_data(self, self.pointcloud_path)
            self.runner.py_logger.info(f"Loading pointcloud points from {path}\n")
            pcl_coords, pcl_normals = load_point_cloud(path)
            pcl_coords = torch.from_numpy(pcl_coords)
            pcl_normals = torch.from_numpy(pcl_normals)
            self.normalize(pcl_coords, pcl_normals)
            self.pcl = torch.cat([pcl_coords, pcl_normals], dim=-1)
        else:
            self.pcl = torch.cat([self._coords, self._normals], dim=-1)

        self._gridPoints = create_grid_points_from_bounds(-self.bbox_size/2,self.bbox_size/2,self.resolution)
        self._gridkdtree = KDTree(self._gridPoints)
        _, self._gridIDs = self._gridkdtree.query(self._coords.numpy())
        self._face_coords, self._faces = get_faces(self._coords,self._normals)

        # generate off_coords
        temp_Ids=np.random.randint(low=0,high=self._coords.shape[0],size=(self.num_off_points,))
        self._off_coords = self._coords[temp_Ids]
        _off_coords_offset = ((torch.rand(self.num_off_points, 3) - 0.5).numpy())
        _off_coords_offset  *= self.bbox_size*15/self.resolution
        self._off_coords += _off_coords_offset
        self._off_coords = self._off_coords.numpy()
        # self._off_coords = (torch.rand(self.num_off_points,3)-0.5).numpy()
        # self._off_coords *= self.bbox_size


        self._gridHasFeature = np.zeros([self.resolution **3,1],dtype=int)
        self._gridHasFeature[self._gridIDs] = 1

        self._gridFeaturePoints = np.zeros([self.resolution ** 3, self.featureSize, 3], dtype=float)
        temp_grid_Ids = np.where(self._gridHasFeature!=0)[0]
        # mark grid where sample coords in it
        for grid_id in temp_grid_Ids:
            query_feature_coords_idcs = np.where(self._gridIDs == grid_id)[0]
            if query_feature_coords_idcs.size != 0:
                if(query_feature_coords_idcs.size>self.featureSize):
                    query_feature_coords_idids = np.random.choice(query_feature_coords_idcs.size,
                                                                  self.featureSize,replace=False)
                    query_feature_coords_idcs = torch.from_numpy(
                        query_feature_coords_idcs[query_feature_coords_idids]).unsqueeze(0)
                else:
                    query_feature_coords_idcs = torch.from_numpy(query_feature_coords_idcs).unsqueeze(0)
                _temp_coords = self._coords[query_feature_coords_idcs].numpy()
                self._gridFeaturePoints[grid_id,:_temp_coords.shape[1],:] = _temp_coords


        # mark grids that voxels has feature
        self._gridVoxelSize = self._gridHasFeature *3
        voxelnum = 3
        while (voxelnum <= self.voxelNum):
            temp_grid_nopoint_Id = np.where(self._gridVoxelSize==0)[0]
            temp_grid_nopoint_VoxelIds=self.Ids2VoxelIds(temp_grid_nopoint_Id,voxelnum)
            self._gridVoxelSize[temp_grid_nopoint_Id] = self._gridHasFeature[temp_grid_nopoint_VoxelIds].max(axis=1)*voxelnum
            voxelnum += 2

        # preget voxel coords
        self._gridVoxelsFeaturePoints = np.zeros([self.resolution ** 3, self.featureSize, 3], dtype=float)
        temp_grid_Voxels_Ids = np.where(self._gridVoxelSize!=0)[0]
        for grid_voxel_id in temp_grid_Voxels_Ids:
            # get voxel ids
            query_voxel_feature_idcs = self.Ids2VoxelIds([grid_voxel_id],self._gridVoxelSize[grid_voxel_id][0])
            # get voxel ids where coords in it
            query_grid_ids = np.where(self._gridHasFeature[query_voxel_feature_idcs]!=0)[1]
            query_grid_ids = query_voxel_feature_idcs[0][query_grid_ids]
            # query feature coords by voxel ids
            query_feature_voxels_coords = self._gridFeaturePoints[query_grid_ids]
            # reshape and remove 0
            temp_query_feature_voxels_coords = query_feature_voxels_coords.reshape(-1,3)
            query_feature_voxels_coords = temp_query_feature_voxels_coords[
                np.where(temp_query_feature_voxels_coords!=0)[0]]
            query_feature_voxels_coords = np.unique(query_feature_voxels_coords,axis=0)
            if(query_feature_voxels_coords.shape[0]>self.featureSize):
                query_feature_coords_idids = np.random.choice(query_feature_voxels_coords.shape[0],
                                                                  self.featureSize,replace=False)
                query_feature_voxels_coords = torch.from_numpy(
                        query_feature_voxels_coords[query_feature_coords_idids])
            self._gridVoxelsFeaturePoints[grid_voxel_id,:query_feature_voxels_coords.shape[0],
                                          :] = query_feature_voxels_coords

        
        # calculate udf
        off_coords_gridIds = self.getGridIds(torch.from_numpy(self._off_coords))
        query_offCoords_Ids = np.where(self._gridVoxelSize[off_coords_gridIds]!=0)[0]
        self._off_udf = cal_off_point_cloud(self._off_coords, query_offCoords_Ids, self._face_coords,self._faces)


        self.runner.logger.log_mesh("pointclouddata", self._coords[None,...], None, vertex_normals=self._normals[None,...])
        self._has_init = True

    def __len__(self):
        self._init()
        return (self.num_points// int(self.batch_size)) + 1

    def __getitem__(self, idx):
        self._init()
        point_cloud_size = self._coords.shape[0]

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        total_samples = self.batch_size
        on_surface_samples = self.batch_size - off_surface_samples
        
        rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
                                                      size=on_surface_samples))
        on_surface_coords = self._coords[rand_idcs, :]
        on_surface_normals = self._normals[rand_idcs, :]


        sdf = torch.zeros((total_samples, 1))  # on-surface = 0

        pointcloud = torch.zeros((self.pointcloud_size, 3))

        if self.pointcloud_size > 0:
            # sample randomly from point cloud
            rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
                                     size=self.pointcloud_size))
            pointcloud = self.pcl[rand_idcs, :]

        if(off_surface_samples > 0):
            rand_off_idcs = torch.from_numpy(np.random.choice(self.num_off_points,
                                         size=off_surface_samples))
            # off_surface_coords = torch.rand(off_surface_samples, 3) - 0.5
            off_surface_coords = torch.from_numpy(self._off_coords[rand_off_idcs])
            off_surface_normals = torch.ones((off_surface_samples, 3)) * -1

        
            sdf[on_surface_samples:, :] = torch.from_numpy(self._off_udf[rand_off_idcs]).unsqueeze(1)

            # sdf[on_surface_samples:, :] = torch.from_numpy(off_surface_udf).unsqueeze(1)

            coords = torch.cat((on_surface_coords, off_surface_coords),
                                    dim=0)
            normals = torch.cat((on_surface_normals, off_surface_normals),
                                    dim=0)
        else:
            coords = on_surface_coords
            normals = on_surface_normals

        # if(idx == 0):
        #     print(idx)

        result = {
                "coords" : coords,
                "normal_out" : normals,
                "sdf_out" : sdf,
                "pointcloud": pointcloud,
        }
        if(self.isEncoding):
            result["featurepoints"], result["relative_coords"], result["query_has_feature"] = self.getFeatureCoords(coords)

        # if(idx == 0):
        #     print("0")
        return result

    def getFeatureCoords(self,_coords):
        if(self.isEncoding):
            #######_coords:tensor[batch,3]
            # get the only center id
            _, grid_idcs = self._gridkdtree.query(_coords.detach().squeeze(0).numpy(),k=1)
            # get the only center
            grid_centers = self._gridPoints[grid_idcs]

            # # query has_feature points 
            query_feature_size = torch.from_numpy(self._gridVoxelSize[grid_idcs])
            # # get voxel centers id
            # _, grid_idcss = self._gridkdtree.query(grid_centers,k=self.voxelNum)
            # get voxel coords
            query_feature_coords = torch.from_numpy(self._gridVoxelsFeaturePoints[grid_idcs])
            # query_feature_coords = torch.from_numpy(self._gridFeaturePoints[grid_idcss])
            # reshape only centers
            _grid_centers = torch.from_numpy(grid_centers).unsqueeze(1)
            # normalize voxel coords
            query_feature_coords = self.normalize_feature(query_feature_coords, self.resolution, _grid_centers)
            # reshape voxel coords
            query_feature_coords = query_feature_coords.reshape([query_feature_coords.shape[0],-1,3])

            # get relative coords
            relative_coords = _coords.detach().squeeze(0)
            relative_coords = self.normalize_feature(relative_coords, self.resolution, grid_centers)
            # cal gradient
            # grad = gradient(proj_coords, _coords)
        else:
            query_feature_coords = None
            relative_coords = None
            query_feature_size = None
        return query_feature_coords, relative_coords, query_feature_size
    
    def getGridIds(self,_coords):
        if(self.isEncoding):
            # get feature voxel coords
            _, query_feature_Ids = self._gridkdtree.query(_coords.detach().squeeze(0).numpy(),k=1)
        else:
            query_feature_Ids = None
        return query_feature_Ids
    
    
    def Ids2VoxelIds(self,Ids,voxelnum):
        if(self.isEncoding):
            # get feature voxel coords
            _, query_feature_Voxel_Ids = self._gridkdtree.query(self._gridPoints[Ids],k=voxelnum**3)
        else:
            query_feature_Voxel_Ids = None
        return query_feature_Voxel_Ids
    
    # get all grid feature coords,to precal voxel features
    def getAllFeatureCoords(self):
        all_feature_coords, _, _ = self.getFeatureCoords(torch.from_numpy(self._gridPoints))
        return all_feature_coords

    def normalize_feature(self, coords, res, centers):
        _centers = centers
        # scale = 1.0/(res-1)
        # normalize feature coords or relative coords
        if(coords.shape[-2]>centers.shape[-2]):
            _centers = _centers.repeat(1,coords.shape[1],1)
            # result = torch.where(coords==0,torch.zeros_like(coords),(coords-_centers)/scale)
            result = torch.where(coords==0,torch.zeros_like(coords),(coords-_centers))
        else:
            result = (coords-_centers)
            # result = (coords-_centers)/scale
            # with torch.enable_grad():
            #     result = (coords-_centers)/scale
            #     grad = gradient(result, coords)
        # with torch.enable_grad():
        #     grad = gradient(result, coords)

        return result.float()
