import os
import sys
sys.path.append("../")
from models.fields import UDFNetwork
# import multiprocessing as mp
from df_dist import distance_count
# from eval_mesh import evaluate_pcs
import trimesh
import open3d as o3d
import json
import numpy as np
import torch
from pyhocon import ConfigFactory
import pymeshlab

ms = pymeshlab.MeshSet()


exp_root = "D:\\mesh"
gt_root = "../data/shapenet_car/input"
score_dict = {}
score_dict_meshudf = {}
score_dict_capudf = {}
bad_results = []
r = 0


def calculate_topology_measure(path):
    ms.load_new_mesh(path)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_vertices()
    re = ms.get_topological_measures()
    ms.delete_current_mesh()
    return re["genus"], re["number_holes"]


def calculate_MLP_distance(mesh, query_fun):
    xyz = torch.from_numpy(np.asarray(mesh.sample_points_uniformly(number_of_points=100000).points).astype(np.float32)).cuda()

    d = query_fun(xyz).mean().detach().cpu().numpy()
    return float(d)


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


def batch_evaluate(name,conf):
    # device = torch.device('cuda')
    # udf_network = UDFNetwork(**conf['model.udf_network']).to(device)
    # checkpoint_name = conf.get_string('evaluate.load_ckpt')
    # checkpoint = torch.load(os.path.join(exp_root,name, 'checkpoints', checkpoint_name),
    #                         map_location=device)
    # print(os.path.join(exp_root,name, 'checkpoints', checkpoint_name))
    # udf_network.load_state_dict(checkpoint['udf_network_fine'])
    # query_fun = lambda pts: udf_network.udf(pts)
    # gt_path = os.path.join(gt_root,"{}.ply".format(name))
    # if not os.path.exists(gt_path):
    #     print("not data {}".format(name))
    #     return
    # gt_mesh = trimesh.load_mesh(gt_path)
    # total_size = (gt_mesh.bounds[1] - gt_mesh.bounds[0]).max()
    # centers = (gt_mesh.bounds[1] + gt_mesh.bounds[0]) / 2
    #
    # gt_mesh.apply_translation(-centers)
    # gt_mesh.apply_scale(1 / total_size)
    # gt_mesh = gt_mesh.as_open3d
    # gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_root,"{}gt_sample.ply".format(name)))
    # if not os.path.exists(os.path.join(exp_root,name,"mesh","{}.ply".format(name))):
    #     print("not cut result {}".format(name))
    #     return
    # result_path_1 = os.path.join(exp_root,name,"mesh","{}_{}_new.ply".format(name,str(r)))
    # result_path_2 = os.path.join(exp_root, name, "mesh", "{}-1_{}_new.ply".format(name, str(r)))
    #if os.path.exists(result_path_1):
    #    our_result_1 = o3d.io.read_triangle_mesh(result_path_1)
    #    our_result_2 = o3d.io.read_triangle_mesh(result_path_2)
    #else:
    #    print(name)
    #    bad_results.append(name)
    #    return
    # if isinstance(gt_mesh, trimesh.PointCloud):
    #     gt_pc = gt_mesh.vertices
    # else:
    #     gt_pc = gt_mesh.sample(100000)
    # meshudf_path = os.path.join(exp_root,name,"mesh", "{}-meshudf_clean.ply".format(name))
    # pred_meshudf = o3d.io.read_triangle_mesh(meshudf_path)
    capudf_path = os.path.join(exp_root,"{}".format(name))
    pred_capudf = o3d.io.read_triangle_mesh(capudf_path)
    # pred_pc = our_result.sample(100000)
    #re_ = distance_count(gt_mesh, our_result_1,100000)
    #re__ = distance_count(gt_mesh, our_result_2, 100000)
    # re_.append(calculate_MLP_distance(our_result_1,query_fun))
    # print(re_[3])
    # re__.append(calculate_MLP_distance(our_result_2,query_fun))
    # re_.append(len(our_result_1.get_non_manifold_vertices()))
    # re__.append(len(our_result_2.get_non_manifold_vertices()))
    # re_.append(len(our_result_1.get_non_manifold_edges()))
    # re__.append(len(our_result_2.get_non_manifold_edges()))
    #g, h = calculate_topology_measure(result_path_1)
    #re_.append(g)
    #re_.append(h)
    # g, h = calculate_topology_measure(result_path_2)
    # re__.append(g)
    # re__.append(h)
    # if re_[2]<re__[2]:
    #     re=re_
    # else:
    #     re=re__
    #if np.asarray(our_result_1.vertices).shape[0]>np.asarray(our_result_2.vertices).shape[0]:
    #    re=re_
    #else:
    #re=re_

    #score_dict[name] = re
    # re2 = distance_count(gt_mesh,pred_meshudf,100000)
    # # re2.append(calculate_MLP_distance(pred_meshudf, query_fun))
    # re2.append(len(pred_meshudf.get_non_manifold_vertices()))
    # re2.append(len(pred_meshudf.get_non_manifold_edges()))
    # g, h = calculate_topology_measure(meshudf_path)
    # re2.append(g)
    # re2.append(h)
    # score_dict_meshudf[name] = re2
    # re3 = distance_count(gt_mesh,pred_capudf,100000)
    # # re3.append(calculate_MLP_distance(pred_capudf, query_fun))
    re3 = []
    re3.append(len(pred_capudf.get_non_manifold_vertices())/len(pred_capudf.vertices)*100)
    # re3.append(len(pred_capudf.get_non_manifold_edges())/len(pred_capudf.edges)*100)
    # g, h = calculate_topology_measure(capudf_path)
    # re3.append(g)
    # re3.append(h)
    score_dict_capudf[name] = re3
    print("***************")
    print(name)
    #print(re)
    # print(re2)
    print(re3)
    print("***************")
    return



def all_test(conf):
    for p in os.listdir(exp_root):
        batch_evaluate(p,conf)
    #with open("recording_{}.json".format(str(r)), "w") as f:
    #    json.dump(score_dict,f,indent=4)
    # with open("recording_meshudf.json", "w") as f:
    #     json.dump(score_dict_meshudf,f,indent=4)
    with open("recording_capudf_non.json", "w") as f:
        json.dump(score_dict_capudf,f,indent=4)
    # with open("recording_bad.json", "w") as f:
    #     json.dump(bad_results,f,indent=4)


def test_batch(test_file,conf):
    with open(test_file, "r") as f:
        paths = json.load(f)
    for p in paths:
        batch_evaluate(p,conf)
    with open("recording_capudf_{}.json".format(os.path.basename(test_file)[:-4]), "w") as f:
        json.dump(score_dict_capudf,f,indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=float)
    parser.add_argument('--batch_file', type=str, default=None)
    parser.add_argument('--conf', type=str, default='../confs/cloth.conf')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    f = open(args.conf)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)
    r = args.r
    if args.batch_file is None:
        all_test(conf)
    else:
        test_batch(args.batch_file,conf)
