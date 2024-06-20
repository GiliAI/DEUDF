import os
import sys
sys.path.append("../")
import trimesh
import open3d as o3d
import json
import numpy as np
import torch
from pyhocon import ConfigFactory
import pymeshlab

ms = pymeshlab.MeshSet()


exp_root = "../experiment/out_cloth_batch"
gt_root = "../data/cloth/data_visualize"
score_dict = {}
score_dict_meshudf = {}
score_dict_capudf = {}
bad_results = []


def calculate_self_intersection(path):
    ms.load_new_mesh(path)
    ms.compute_selection_by_self_intersections_per_face()
    faces = ms.current_mesh().face_selection_array()
    ms.delete_current_mesh()
    return np.sum(faces)


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
    if not os.path.exists(os.path.join(gt_root,"{}gt_sample.ply".format(name))):
        print("not train {}".format(name))
        return
    result_path_1 = os.path.join(exp_root,name,"mesh","{}-0_0.005_new.ply".format(name))
    result_path_2 = os.path.join(exp_root, name, "mesh", "{}-1_0.005_new.ply".format(name))
    if not os.path.exists(result_path_1):
        result_path_1 = os.path.join(exp_root, name, "mesh", "{}-0_0.005.ply".format(name))
        result_path_2 = os.path.join(exp_root, name, "mesh", "{}-1_0.005.ply".format(name))
    if os.path.exists(result_path_1):
        our_result_1 = o3d.io.read_triangle_mesh(result_path_1)
        our_result_2 = o3d.io.read_triangle_mesh(result_path_2)
    else:
        print(name)
        bad_results.append(name)
        return
    # if isinstance(gt_mesh, trimesh.PointCloud):
    #     gt_pc = gt_mesh.vertices
    # else:
    #     gt_pc = gt_mesh.sample(100000)
    # pred_pc = our_result.sample(100000)
    # re_ = distance_count(gt_mesh, our_result_1,100000)
    # re__ = distance_count(gt_mesh, our_result_2, 100000)
    # re_.append(calculate_MLP_distance(our_result_1,query_fun))
    # print(re_[3])
    # re__.append(calculate_MLP_distance(our_result_2,query_fun))
    # if re_[2]<re__[2]:
    #     re=re_
    # else:
    #     re=re__
    if np.asarray(our_result_1.vertices).shape[0]>np.asarray(our_result_2.vertices).shape[0]:
        re=[calculate_self_intersection(result_path_1)/len(our_result_1.triangles) *100]
    else:
        re=[calculate_self_intersection(result_path_2)/len(our_result_2.triangles) *100]

    score_dict[name] = re
    print("***************")
    print(name)
    print(re)
    # print(re2)
    # print(re3)
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
    with open("recording_self_intersection.json", "w") as f:
        json.dump(score_dict,f,indent=4)


def test_batch(test_file,conf):
    with open(test_file, "r") as f:
        paths = json.load(f)
    for p in paths:
        batch_evaluate(p,conf)
    # _sum = 0
    # for d in score_dict.values():
    #     _sum += d[2]*10000
    # if len(score_dict.keys()) > 0:
    #     print(_sum/len(score_dict.keys()))
    # else:
    #     print("nan")
    # _sum = 0
    # for d in score_dict_meshudf.values():
    #     _sum += d[2]*10000
    # if len(score_dict_meshudf.keys())>0:
    #     print(_sum/len(score_dict_meshudf.keys()))
    # else:
    #     print("nan")
    # _sum = 0
    # for d in score_dict_capudf.values():
    #     _sum += d[2]*10000
    # if len(score_dict_capudf.keys()) >0:
    #     print(_sum/len(score_dict_capudf.keys()))
    # else:
    #     print("nan")
    with open("recording_{}.json".format(os.path.basename(test_file)[:-4]), "w") as f:
        json.dump(score_dict,f,indent=4)
    # with open("recording_capudf_{}.json".format(os.path.basename(test_file)[:-4]), "w") as f:
    #     json.dump(score_dict_capudf,f,indent=4)
    # with open("recording_bad_{}.json".format(os.path.basename(test_file)[:-4]), "w") as f:
    #     json.dump(bad_results,f,indent=4)


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
