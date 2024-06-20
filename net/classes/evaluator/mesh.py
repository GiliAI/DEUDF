import sys
from DCUDF.dcudf.mesh_extraction import dcudf
from DCUDF.dcudf.A_evaluate_custom_test import Evaluator as Extractor
from typing import Dict
import numpy as np
import torch
from evaluator.evaluator import Evaluator
from .helper import get_surface_high_res_mesh
from task.chamfer import Compute_Chamfer,show_Sigmas
import trimesh
from evaluator.optimize_A import getMeshUDF

def calUdf(query_func,input_points,_batch_size):
    all_input_points = input_points.reshape(-1,3)
    result_coords_df = np.zeros([all_input_points.shape[0],4])
    start = 0
    while(start<result_coords_df.shape[0]):
        sample_input_points = all_input_points[start:min(result_coords_df.shape[0],start+_batch_size),:]
        result_coords_df[start:min(result_coords_df.shape[0],start+_batch_size),:3] = sample_input_points
        df = query_func(torch.from_numpy(sample_input_points).float().cuda())
        result_coords_df[start:min(result_coords_df.shape[0],start+_batch_size),-1:] = df.detach().cpu().numpy()
        start += _batch_size
    
    return result_coords_df

def getNewCoords(query_func,input_points,_batch_size):
    all_input_points = input_points.detach().reshape(-1,3)
    result_coords = torch.zeros_like(all_input_points)
    start = 0
    while(start<result_coords.shape[0]):
        sample_input_points = all_input_points[start:min(result_coords.shape[0],start+_batch_size),:]
        new_coords = query_func(sample_input_points.float().cuda())
        result_coords[start:min(result_coords.shape[0],start+_batch_size),:] = new_coords
        start += _batch_size
    
    return result_coords



class Mesh(Evaluator):

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
        self.threshold : float = 0.005
        self.another : bool = False
        self.isCut : bool = True
        super().__init__(config)
        self.attributes.append(self.attribute)

    def evaluate_mesh_value(self, data=None):
        try:
            bbox_size = getattr(self.runner.data, 'bbox_size', self.bbox_size)
        except:
            bbox_size = self.bbox_size

        try:
            self.runner.network.iniFeatures()
        except Exception as e:
            print(e)
            print("init Feature failed,if no features,ignore this message")


        if data is not None:
            # fea = self.encode_network(data)
            data.pop('coords', None)
            data.pop('featurepoints', None)

        extractor = dcudf(lambda x: self.evaluate_network(x, fea=self.encode_network(x), **data)[self.attribute],
                              self.resolution,self.threshold,bound_min=[-1,-1,-1],bound_max=[1,1,1],
                              is_cut = self.isCut,region_rate=10, laplacian_weight=50)
        if self.another:
            dcMesh, mesh, another_mesh = extractor.optimize()
        else :
            dcMesh, mesh, _ = extractor.optimize()


        meshUDF = getMeshUDF(lambda x: self.evaluate_network(x, fea=self.encode_network(x), **data)[self.attribute],
                             self.resolution)
        try:
            if not mesh.is_empty:
                if self.another:
                    return np.array(mesh.vertices), np.array(mesh.faces), np.array(mesh.vertex_normals),\
                        np.array(dcMesh.vertices), np.array(dcMesh.faces), np.array(dcMesh.vertex_normals),\
                        np.array(meshUDF.vertices), np.array(meshUDF.faces), np.array(meshUDF.vertex_normals),\
                        np.array(another_mesh.vertices), np.array(another_mesh.faces), np.array(another_mesh.vertex_normals)
                else :
                    return np.array(mesh.vertices), np.array(mesh.faces), np.array(mesh.vertex_normals),\
                            np.array(dcMesh.vertices), np.array(dcMesh.faces), np.array(dcMesh.vertex_normals),\
                            np.array(meshUDF.vertices), np.array(meshUDF.faces), np.array(meshUDF.vertex_normals),\
                            np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float)
            return np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),\
                np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float)
        except Exception as e:
            tb = sys.exc_info()[2]
            self.runner.py_logger.error(repr(e))

            return np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),\
                np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),\
                np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),\
                np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float),np.array([[0,0,0]],dtype=float)





    def epoch_hook(self, epoch, data: Dict =None):
        if (epoch % self.frequency == 0 and (epoch <= self.stop_after or self.stop_after == 0)):
            if ("Train" in self.runner.active_task.__class__.__name__ ) and self.skip_first and epoch == 0:
                return
            self.runner.py_logger.info(f"Generating {self.name} with resolution {self.resolution}")
            verts, faces, _, dcVerts, dcFaces, _, meshUDFverts,meshUDFfaces, _,aVerts,aFaces,_= self.evaluate_mesh_value(data)
            if(np.all(verts == 0) and np.all(faces == 0)):
                self.runner.py_logger.info(f"bad udf,no mesh generated")
                return

            # if(self.compute_chamfer):
            #     if(self.another):
            #         Compute_Chamfer(trimesh.Trimesh(aVerts.astype(np.float32),aFaces),self.runner,f"{self.name}_Achamfer_{{0}}_{{1}}", True)
            #         Compute_Chamfer(trimesh.Trimesh(verts.astype(np.float32),faces),self.runner,f"{self.name}_chamfer_{{0}}_{{1}}", True)
            #         Compute_Chamfer(trimesh.Trimesh(meshUDFverts.astype(np.float32),meshUDFfaces),self.runner,f"{self.name}_Mchamfer_{{0}}_{{1}}", True)
            #     else:
            #         Compute_Chamfer(trimesh.Trimesh(verts.astype(np.float32),faces),self.runner,f"{self.name}_chamfer_{{0}}_{{1}}", True)
            #         Compute_Chamfer(trimesh.Trimesh(meshUDFverts.astype(np.float32),meshUDFfaces),self.runner,f"{self.name}_Mchamfer_{{0}}_{{1}}", True)


            self.runner.logger.log_mesh(self.name, verts.reshape([1, -1, 3]),
                                        faces.reshape([1, -1, 3]))

            self.runner.logger.log_mesh(self.name+"DC", dcVerts.reshape([1, -1, 3]),
                                        dcFaces.reshape([1, -1, 3]))
            self.runner.logger.log_mesh(self.name+"MESHUDF", meshUDFverts.reshape([1, -1, 3]),
                                        meshUDFfaces.reshape([1, -1, 3]))
            if self.another:
                self.runner.logger.log_mesh(self.name+"Another", aVerts.reshape([1, -1, 3]),
                                        aFaces.reshape([1, -1, 3]))