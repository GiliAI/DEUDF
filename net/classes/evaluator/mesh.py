import sys
# sys.path.append('/home/xz/reposotories/Unsigned_Marching_Cubes-1.0')
# from evaluate_custom import Evaluator as Evaluator_ours
# from Unsigned_Marching_Cubes.evaluate_custom import Evaluator as Evaluator_ours
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
from DualMeshUDF import extract_mesh

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
        self.isCut : bool = False
        self.dmudf : bool = False
        self.meshudf : bool = False
        super().__init__(config)
        self.attributes.append(self.attribute)

    def query_func(self, x, pd=True, **kwargs):
        target_shape = list(x.shape)
        pts = x.reshape(-1, 3)
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).cuda()
        input = pts.reshape(-1, pts.shape[-1]).float()
        d = self.evaluate_network(input, fea=None, **kwargs)[self.attribute]
        target_shape[-1] = 1
        d = d.reshape(target_shape)
        if pd:
            d = d.detach().cpu().numpy()
        
        return d
    def grad_func(self, x, **kwargs):
        
        target_shape = list(x.shape)

        x = torch.tensor(x)
        x.requires_grad_(True)
        input = x.reshape(-1, x.shape[-1]).float().cuda()
        y = self.query_func(input, pd=False, **kwargs)

        y.requires_grad_(True)
        # udf_p = y.detach()

        d_output = torch.ones_like(y, requires_grad=True, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        
        grad_p = gradients.reshape(target_shape).detach().cpu().numpy()
        target_shape[-1] = 1
        udf_p = y.reshape(target_shape).detach().cpu().numpy()
        

        return udf_p, grad_p





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
                            #   ,optimize_query_func=lambda x: self.evaluate_network(x, fea=self.encode_network(x), **optimize_data)[self.attribute])
        dcMesh, mesh, another_mesh = extractor.optimize()
        

        if not self.another:
            another_mesh = None
        if self.dmudf:
            # mesh_v, mesh_f =extract_mesh_from_udf(lambda x: self.evaluate_network(x, fea=self.encode_network(x), **data)[self.attribute], device)
            mesh_v, mesh_f = extract_mesh(lambda x: self.query_func(x, **data), lambda x: self.grad_func(x, **data))
            dmudf_mesh = trimesh.Trimesh(vertices = mesh_v, faces = mesh_f)
        else:
            dmudf_mesh = None
        

        if self.meshudf:
            meshUDF = getMeshUDF(lambda x: self.evaluate_network(x, fea=self.encode_network(x), **data)[self.attribute],
                                self.resolution)
        else:
            meshUDF = None
        # meshUDF = None
        try:
            if not mesh.is_empty:
                return mesh, dcMesh, meshUDF, another_mesh, dmudf_mesh
            else:
                return None

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
            #this is plain wrong
            #with torch.no_grad():
            meshName = ["", "DC", "MESHUDF", "Another", "DMUDF"]
            meshes = self.evaluate_mesh_value(data)
            for i, mesh in enumerate(meshes):
                if mesh is None:
                    self.runner.py_logger.info(f"bad udf,no mesh generated")
                    continue
                self.runner.logger.log_mesh(self.name+meshName[i], mesh.vertices.reshape([1, -1, 3]),
                                        mesh.faces.reshape([1, -1, 3]))

