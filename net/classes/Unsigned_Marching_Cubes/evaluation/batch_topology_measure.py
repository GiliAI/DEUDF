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


exp_root = "../experiment/out_cloth_batch"
gt_root = "../data/cloth/data_visualize"
score_dict = {}
score_dict_meshudf = {}
score_dict_capudf = {}
bad_results = []


def calculate_MLP_distance(mesh, query_fun):
    xyz = torch.from_numpy(np.asarray(mesh.sample_points_uniformly(number_of_points=100000).points).astype(np.float32)).cuda()

    d = query_fun(xyz)
    print(d.shape)
    return d.data.cpu().numpy()[0]


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
    if not os.path.exists(os.path.join(gt_root,"{}gt_sample.ply".format(name))):
        print("not train {}".format(name))
        return
    gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_root,"{}gt_sample.ply".format(name)))
    # if not os.path.exists(os.path.join(exp_root,name,"mesh","{}.ply".format(name))):
    #     print("not cut result {}".format(name))
    #     return
    result_path_1 = os.path.join(exp_root,name,"mesh","{}-0.ply".format(name))
    result_path_2 = os.path.join(exp_root, name, "mesh", "{}-1.ply".format(name))
    result_path_3 = os.path.join(exp_root,name,"mesh","{}-0_merge.ply".format(name))
    result_path_4 = os.path.join(exp_root, name, "mesh", "{}-1_merge.ply".format(name))
    if os.path.exists(result_path_1):
        our_result_1 = o3d.io.read_triangle_mesh(result_path_1)
        our_result_2 = o3d.io.read_triangle_mesh(result_path_2)
        our_result_3 = o3d.io.read_triangle_mesh(result_path_3)
        our_result_4 = o3d.io.read_triangle_mesh(result_path_4)
    else:
        print(name)
        bad_results.append(name)
        return
    # if isinstance(gt_mesh, trimesh.PointCloud):
    #     gt_pc = gt_mesh.vertices
    # else:
    #     gt_pc = gt_mesh.sample(100000)
    pred_meshudf = o3d.io.read_triangle_mesh(os.path.join(exp_root,name,"mesh", "{}-meshudf.ply".format(name)))
    pred_capudf = o3d.io.read_triangle_mesh(os.path.join(exp_root,name,"mesh", "0_mesh.obj"))
    # pred_pc = our_result.sample(100000)
    re_ = distance_count(gt_mesh, our_result_1,100000)
    re__ = distance_count(gt_mesh, our_result_2, 100000)
    # re_.append(calculate_MLP_distance(our_result_1,query_fun))
    # print(re_[3])
    # re__.append(calculate_MLP_distance(our_result_2,query_fun))
    re_.append(len(our_result_1.get_non_manifold_vertices()))
    re__.append(len(our_result_2.get_non_manifold_vertices()))
    re_.append(len(our_result_3.get_non_manifold_vertices()))
    re__.append(len(our_result_4.get_non_manifold_vertices()))
    re_.append(len(our_result_1.get_non_manifold_edges()))
    re__.append(len(our_result_2.get_non_manifold_edges()))
    re_.append(len(our_result_3.get_non_manifold_edges()))
    re__.append(len(our_result_4.get_non_manifold_edges()))
    # if re_[2]<re__[2]:
    #     re=re_
    # else:
    #     re=re__
    if np.asarray(our_result_1.vertices).shape[0]>np.asarray(our_result_2.vertices).shape[0]:
        re=re_
    else:
        re=re__

    score_dict[name] = re
    re2 = distance_count(gt_mesh,pred_meshudf,100000)
    # re2.append(calculate_MLP_distance(pred_meshudf, query_fun))
    re2.append(len(pred_meshudf.get_non_manifold_vertices()))
    re2.append(len(pred_meshudf.get_non_manifold_edges()))
    score_dict_meshudf[name] = re2
    re3 = distance_count(gt_mesh,pred_capudf,100000)
    # re3.append(calculate_MLP_distance(pred_capudf, query_fun))
    re3.append(len(pred_capudf.get_non_manifold_vertices()))
    re3.append(len(pred_capudf.get_non_manifold_edges()))
    score_dict_capudf[name] = re3
    print("***************")
    print(name)
    print(re[3])
    print(re2[3])
    print(re3[3])
    print(re[4])
    print(re2[4])
    print(re3[4])
    print("***************")
    return



def all_test(conf):
    for p in os.listdir(exp_root):
        batch_evaluate(p,conf)
    _sum = 0
    for d in score_dict.values():
        _sum += d[1]*10000
    if len(score_dict.keys()) > 0:
        print(_sum/len(score_dict.keys()))
    else:
        print("nan")
    _sum = 0
    for d in score_dict_meshudf.values():
        _sum += d[1]*10000
    if len(score_dict_meshudf.keys())>0:
        print(_sum/len(score_dict_meshudf.keys()))
    else:
        print("nan")
    _sum = 0
    for d in score_dict_capudf.values():
        _sum += d[1]*10000
    if len(score_dict_capudf.keys()) >0:
        print(_sum/len(score_dict_capudf.keys()))
    else:
        print("nan")
    with open("recording.json", "w") as f:
        json.dump(score_dict,f,indent=4)
    with open("recording_meshudf.json", "w") as f:
        json.dump(score_dict_meshudf,f,indent=4)
    with open("recording_capudf.json", "w") as f:
        json.dump(score_dict_capudf,f,indent=4)
    with open("recording_bad.json", "w") as f:
        json.dump(bad_results,f,indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_file', type=str, default=None)
    parser.add_argument('--conf', type=str, default='../confs/cloth.conf')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    f = open(args.conf)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)
    if args.batch_file is None:
        all_test(conf)
    else:
        test_batch(args.batch_file,conf)
