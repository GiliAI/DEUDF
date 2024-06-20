import torch
import numpy as np
import igl

from helper import get_path_for_data
from data.dataset import CDataset
from network.network import Network
from pykdtree.kdtree import KDTree
from typing import Dict
import pymeshlab

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
    mesh = ms.current_mesh()

    coords = mesh.vertex_matrix().astype('float32')
    normals = mesh.vertex_normal_matrix().astype('float32')
    # addition
    # ms.generate_surface_reconstruction_screened_poisson()
    # mesh = ms.current_mesh()
    # addition
    # faces = mesh.face_matrix()
    # face_coords = mesh.vertex_matrix().astype('float32')

    if num_points < coords.shape[0]:
        idx = np.random.permutation(coords.shape[0])[:num_points]
        coords = np.ascontiguousarray(coords[idx])
        normals = np.ascontiguousarray(normals[idx])
        # addition
        # faces = np.ascontiguousarray(faces[idx])
    # addition
    # return coords, normals, faces

    # return coords, normals, face_coords, faces
    return coords, normals

def get_faces(_coords,_normals):
    m = pymeshlab.Mesh(vertex_matrix=_coords, v_normals_matrix=_normals)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "temp")
    ms.generate_surface_reconstruction_screened_poisson()
    mesh = ms.current_mesh()

    return mesh.vertex_matrix(),mesh.face_matrix()
def cal_off_point_cloud(query_coords,_coords,_faces):
    # addition
    # faces = mesh.face_matrix()
    # m = pymeshlab.Mesh(vertex_matrix=_coords,face_matrix=_faces)
    # ms = pymeshlab.MeshSet()
    # ms.add_mesh(m, "temp")
    # mesh = ms.current_mesh()

    udf = np.abs(igl.signed_distance(query_coords,_coords, _faces)[0])

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
        self.res = 4
        # self._offcoords : np.ndarray = None
        self._gridkdtree = None
        self._gridFeaturePoints : np.ndarray = None
        # self._offkdtree = None
        # self._onkdtree = None
        # self._udf : np.ndarray = None
        # self.getNearestPC : bool = False

        self.isEncoding : bool = True
        self.featureSize : int = 256

        super().__init__(config)
        if self.pointcloud_path is None:
            self.pointcloud_path = self.path

    def _init(self):
        if(self._has_init):
            return
        path = get_path_for_data(self, self.path)
        self.runner.py_logger.info(f"Loading pointcloud from {path}\n")
        # addition
        # self._coords, self._normals, self._face_coords, self._faces = load_point_cloud(path, self.num_points)
        # self._faces = torch.from_numpy(self._faces)

        self._coords, self._normals = load_point_cloud(path, self.num_points)

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

        self._gridPoints = create_grid_points_from_bounds(-self.bbox_size/2,self.bbox_size/2,self.res)
        self._gridkdtree = KDTree(self._gridPoints)
        _, self._gridIDs = self._gridkdtree.query(self._coords.numpy())
        self._face_coords, self._faces = get_faces(self._coords,self._normals)

        for grid_id in range(self.res ** 3):
            query_feature_coords_idcs = np.where(self._gridIDs == grid_id)[0]
            if query_feature_coords_idcs.size == 0:
                temp_target_fea_coords = np.ones([1, self.featureSize, 3], dtype=int) * -1
            else:
                query_feature_coords_idids = np.random.randint(0, query_feature_coords_idcs.size,
                                                               size=self.featureSize)
                query_feature_coords_idcs = torch.from_numpy(
                    query_feature_coords_idcs[query_feature_coords_idids]).unsqueeze(0)
                temp_target_fea_coords = self._coords[query_feature_coords_idcs].numpy()
            if self._gridFeaturePoints is None:
                self._gridFeaturePoints = temp_target_fea_coords
            else:
                self._gridFeaturePoints = np.concatenate((self._gridFeaturePoints, temp_target_fea_coords), axis=0)

        # addition
        # self._onkdtree = KDTree(self._coords.numpy())
        # init offsamples
        # self._offcoords = torch.rand(self.num_points, 3) - 0.5
        # self._offcoords *= self.bbox_size

        # self._offkdtree = KDTree(self._offcoords.numpy())
        # self._udf = np.abs(igl.signed_distance(self._offcoords.numpy(), self._face_coords, self._faces)[0])
        # self._udf = cal_off_point_cloud(self._coords, self._normals, self._offcoords)



        self.runner.logger.log_mesh("pointclouddata", self._coords[None,...], None, vertex_normals=self._normals[None,...])
        self._has_init = True

    def __len__(self):
        self._init()
        return (self._coords.shape[0] // int(self.batch_size*(1-self.factor_off_surface))) + 1

    def __getitem__(self, idx):
        self._init()
        point_cloud_size = self._coords.shape[0]

        off_surface_samples = int(self.batch_size  * self.factor_off_surface)
        total_samples = self.batch_size
        on_surface_samples = self.batch_size - off_surface_samples
        # Random coords
        # rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
        #                              size=on_surface_samples))
        #
        # on_surface_coords = self._coords[rand_idcs, :]
        # on_surface_normals = self._normals[rand_idcs, :]

        # addition
        # if(self.getNearestPC):
        #     rand_id = np.random.random_integers(0, point_cloud_size)
        #     _, rand_idcs = self._onkdtree.query(self._coords[rand_id, :].unsqueeze(0).numpy(), k=on_surface_samples)
        #     rand_idcs = torch.from_numpy(rand_idcs.squeeze().astype("long"))
        # else:
        #     rand_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
        #                              size=on_surface_samples))
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
            off_surface_coords = torch.rand(off_surface_samples, 3) - 0.5
            off_surface_normals = torch.ones((off_surface_samples, 3)) * -1

            off_surface_coords *= self.bbox_size
            # off_surface_coords = torch.rand(off_surface_samples, 3) - 0.5
            # if (self.getNearestPC):
            #     _, off_surface_idcs = self._offkdtree.query(self._coords[rand_id, :].unsqueeze(0).numpy(),
            #                                                 k=off_surface_samples)
            #     off_surface_idcs = torch.from_numpy(off_surface_idcs.squeeze().astype("long"))
            # else:
            #     off_surface_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
            #                                                   size=on_surface_samples))
            # off_surface_idcs = torch.from_numpy(np.random.choice(point_cloud_size,
            #                                                      size=on_surface_samples))
            # off_surface_normals = torch.ones((off_surface_samples, 3)) * -1


            # addition
            # off_surface_coords = self._offcoords[off_surface_idcs, :]
            # off_surface_udf = self._udf[off_surface_idcs]
            # off_surface_udf = \
            #     cal_off_point_cloud(on_surface_coords, on_surface_normals, off_surface_coords)


            # addition
            # sdftemp=torch.ones((off_surface_samples,1))
            # sdftemp=np.abs(igl.signed_distance(off_surface_coords,self._coords,))

            # addition
            # sdf[on_surface_samples:, :] = -1  # off-surface = -1
            sdf[on_surface_samples:, :] = torch.from_numpy(cal_off_point_cloud(off_surface_coords.numpy(),
                                                                               self._face_coords,self._faces)).unsqueeze(1)

            # sdf[on_surface_samples:, :] = torch.from_numpy(off_surface_udf).unsqueeze(1)

            coords = torch.cat((on_surface_coords, off_surface_coords),
                                    dim=0)
            normals = torch.cat((on_surface_normals, off_surface_normals),
                                    dim=0)

        else:
            coords = on_surface_coords
            normals = on_surface_normals

        if(self.isEncoding):
            _, grid_idcs = self._gridkdtree.query(coords.numpy())
            # _, coords_on_idcs = self._onkdtree.query(coords.numpy())
            # _, coords_idcs = self._onkdtree.query(self._coords[coords_on_idcs.astype("int64")].numpy(), k=self.featureSize)
            # gridIDs = torch.from_numpy(self._gridIDs.astype("int64")).unsqueeze(1)
            query_feature_coords = torch.from_numpy(self._gridFeaturePoints[grid_idcs])

            # gridIDs = self._gridIDs.astype("int64")
            # query_feature_coords = None
            # for grid_id in grid_idcs:
            #     query_feature_coords_idcs = np.where(gridIDs == grid_id)[0]
            #     if query_feature_coords_idcs.size==0:
            #         temp_target_fea_coords =torch.ones([1,self.featureSize,3],dtype=int)*-1
            #     else:
            #         query_feature_coords_idids = np.random.randint(0, query_feature_coords_idcs.size,
            #                                                        size=self.featureSize)
            #         query_feature_coords_idcs = torch.from_numpy(
            #             query_feature_coords_idcs[query_feature_coords_idids]).unsqueeze(0)
            #         temp_target_fea_coords = self._coords[query_feature_coords_idcs]
            #     if query_feature_coords is None:
            #         query_feature_coords = temp_target_fea_coords
            #     else:
            #         query_feature_coords = torch.cat((query_feature_coords, temp_target_fea_coords),dim=0)

        return {
                "coords" : coords,
                "normal_out" : normals,
                "sdf_out" : sdf,
                "pointcloud": pointcloud,
                "featurepoints": query_feature_coords,
        }

