import argparse
import sys
import numpy as np
import torch
from mesh_evaluator import MeshEvaluator
import trimesh
import os
sys.path.append('tools')
from pyhocon import ConfigFactory

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

gt_cache={}

def evaluate_mesh(gt_mesh, es_mesh, n_points=100000, normalize=False):
    # gt_mesh = as_mesh(gt_mesh)
    if isinstance(gt_mesh, trimesh.Trimesh):
        gt_pointcloud, idx = gt_mesh.sample(n_points, return_index=True)
        gt_pointcloud = gt_pointcloud.astype(np.float32)
        gt_normals = gt_mesh.face_normals[idx]
    elif isinstance(gt_mesh, trimesh.PointCloud):
        gt_pointcloud = gt_mesh.vertices.astype(np.float32)
        # np.random.shuffle(gt_pointcloud)
        # gt_pointcloud = gt_pointcloud[:n_points]
        gt_normals = None
    else:
        raise RuntimeError('Unknown data type!')
        # normalize according to the scale of the gt point cloud
    #     if normalize:
    #         center = gt_pointcloud.mean(0)
    #         gt_scale = np.abs(gt_pointcloud-center).max()
    #         print('Normalize with scale: %.2f' % gt_scale)
    #     gt_pointcloud /= gt_scale
    #     gt_cache.update({gt_path: (gt_pointcloud, gt_normals, gt_scale)})
    #
    # total_size = (gt_mesh.bounds[1] - gt_mesh.bounds[0]).max()
    # centers = (gt_mesh.bounds[1] + gt_mesh.bounds[0]) / 2
    # es_mesh.apply_scale(total_size)
    # es_mesh.apply_translation(centers)

    thresholds= [0.005, 0.01]
    evaluator = MeshEvaluator(n_points=n_points)
    result=evaluator.eval_mesh(es_mesh, gt_pointcloud, gt_normals, thresholds=thresholds)

    # print("chamferl1:%.3f"%(result['chamfer-L1']*1000))
    # print("f-score-0.005:%.3f"%(result['f-score-0.005']))
    # print("f-score-0.01:%.3f"%(result['f-score-0.01']))

    return result


def evaluate_pcs(gt_pointcloud, es_pcs, n_points=100000, normalize=False):

    thresholds= [0.005, 0.01]
    evaluator = MeshEvaluator(n_points=100000)
    result=evaluator.eval_pointcloud(es_pcs, gt_pointcloud, thresholds=thresholds)

    # print("chamferl2:%.3f"%(result['chamfer-L2']*10000))
    # print("f-score-0.005:%.3f"%(result['f-score-0.005']))
    # print("f-score-0.01:%.3f"%(result['f-score-0.01']))

    return result


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', type=str, default='./confs/ndf.conf')
    # parser.add_argument('--mcube_resolution', type=int, default=128)
    # parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--dir', type=str, default='test')
    # parser.add_argument('--dataname', type=str, default='demo')
    # parser.add_argument('--pred_mesh', type=str, default='demo')
    #
    # args = parser.parse_args()
    # conf_path = args.conf
    # f = open(conf_path)
    # conf_text = f.read()
    # f.close()
    #
    # conf = ConfigFactory.parse_string(conf_text)

    #
    # dataname_split = args.dataname.split('_', 1)[0]
    #
    # gt_file = os.path.join(conf.get_string('dataset.data_dir'), 'ground_truth', dataname_split+'.obj')
    # pred_file = os.path.join(conf.get_string('general.base_exp_dir'), args.dir, 'mesh', str(conf.get_int('train.step2_maxiter'))+'_mesh.obj')
    # gt_file_dir =
    # pred_file_dir =
    # pred_file_dir = "D:\\Projects\\CXH\\Unsigned_Marching_Cubes\\experiment\\select_mesh"
    # pred_file = "D:\\idf-main\\experiment\\outs_batch\\test\\a_meshudf.ply"
    # gt_file = "D:\\render_data\\all_data_for_render\\high-res-sdf\\dragon-gt.ply"
    # pred_file = "D:\\render_data\\all_data_for_render\\high-res-sdf\\dragon-gt.ply"
    gt_file = "D:\\Projects\\CXH\\Unsigned_Marching_Cubes\\data\\shapenetCars\\e4bc9cf3aaff441f19d740e0f6a9a113_gt_sample.ply"
    pred_file = "D:\\Projects\\CXH\\Unsigned_Marching_Cubes\\data\\shapenetCars\\e4bc9cf3aaff441f19d740e0f6a9a113-meshudf.ply"
    gt_mesh = trimesh.load_mesh(gt_file).vertices
    pred_mesh = trimesh.load_mesh(pred_file).sample(100000)
    print(evaluate_pcs(gt_mesh, pred_mesh, 100000))
    # mark = []
    # for data in os.listdir(gt_file_dir):
    #     gt_file = os.path.join(gt_file_dir,data)
    #     pred_file = os.path.join(pred_file_dir,data)
    #     mark.append(evaluate_mesh(gt_file, pred_file, 8000000))
    # ave_cd = 0
    # for d in mark:
    #     ave_cd += d['chamfer-L2'] * 10000
    # ave_cd /= len(mark)
    # print("ave CD: {}".format(ave_cd))