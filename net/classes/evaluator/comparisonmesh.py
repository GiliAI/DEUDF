import sys
# sys.path.append('/home/xz/reposotories/Unsigned_Marching_Cubes-1.0')
# from evaluate_custom import Evaluator as Evaluator_ours
from Unsigned_Marching_Cubes.evaluate_custom import Evaluator as Evaluator_ours
from typing import Dict
import numpy as np
import torch
from evaluator.evaluator import Evaluator
from .helper import get_surface_high_res_mesh
from task.chamfer import Compute_Chamfer
import trimesh
import igl
import pymeshlab
import os



class ComparisonMesh(Evaluator):

    def __init__(self, config):
        self.batch_size = 100000
        self.frequency =  10
        self.resolution = 100
        self.offset = 0
        self.skip_first :bool = True
        self.bbox_size : float = 2.0
        self.attribute : str = "sdf"
        self.stop_after : int = 0
        self.compute_chamfer : bool = False
        super().__init__(config)
        self.attributes.append(self.attribute)

    def evaluate_mesh_value(self):
        try:
            bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        except:
            bbox_size = self.bbox_size

        _coords, _normals = self.runner.data._coords, self.runner.data._normals
        face_coords, faces = get_faces(_coords, _normals)
        evaluator = Evaluator_ours("idf_udf", lambda x: cal_off_point_cloud(x, _coords=face_coords,_faces=faces),
                                       conf_path="confs/base_idf.conf",
                                       bound_min=[-1, -1, -1], bound_max=[1, 1, 1])
        mesh = evaluator.evaluate()

        try:


            if not mesh.is_empty:
                return np.array(mesh.vertices), np.array(mesh.faces), np.array(mesh.vertex_normals)
            return np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float)
            # addition
            # return np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float),np.array([[0,0,0]],dtype=np.float)
        except Exception as e:
            tb = sys.exc_info()[2]
            self.runner.py_logger.error(repr(e))

            return np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float)


    def epoch_hook(self, epoch):
        if (epoch % self.frequency == 0 and (epoch <= self.stop_after or self.stop_after == 0)):
            if self.runner.active_task.__class__.__name__ == "Train" and self.skip_first and epoch == 0:
                return
            self.runner.py_logger.info(f"Generating {self.name} with resolution {self.resolution}")
            #this is plain wrong
            #with torch.no_grad():
            verts, faces, _ = self.evaluate_mesh_value()

            if(self.compute_chamfer):
                Compute_Chamfer(trimesh.Trimesh(verts.astype(np.float32),faces),self.runner,f"{self.name}_chamfer_{{0}}_{{1}}", True)
            self.runner.logger.log_mesh(self.name, verts.reshape([1, -1, 3]),
                                        faces.reshape([1, -1, 3]))

    def load_point_cloud(path, num_points):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        mesh = ms.current_mesh()

        coords = mesh.vertex_matrix().astype('float32')
        normals = mesh.vertex_normal_matrix().astype('float32')

        if num_points < coords.shape[0]:
            idx = np.random.permutation(coords.shape[0])[:num_points]
            coords = np.ascontiguousarray(coords[idx])
            normals = np.ascontiguousarray(normals[idx])

        return coords, normals


def get_faces(_coords, _normals):
    m = pymeshlab.Mesh(vertex_matrix=_coords, v_normals_matrix=_normals)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "temp")
    ms.generate_surface_reconstruction_screened_poisson()
    mesh = ms.current_mesh()

    return mesh.vertex_matrix(), mesh.face_matrix()

def cal_off_point_cloud(query_coords, _coords, _faces):
        # addition
        # faces = mesh.face_matrix()
        # m = pymeshlab.Mesh(vertex_matrix=_coords,face_matrix=_faces)
        # ms = pymeshlab.MeshSet()
        # ms.add_mesh(m, "temp")
        # mesh = ms.current_mesh()
    _query_coords = query_coords.detach().cpu().numpy()
    udf = np.abs(igl.signed_distance(_query_coords, _coords, _faces)[0])
    udf = torch.from_numpy(udf).cuda()

    return udf