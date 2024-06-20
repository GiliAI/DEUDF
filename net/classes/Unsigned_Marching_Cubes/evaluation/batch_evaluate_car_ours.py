import os
# import multiprocessing as mp
from df_dist import distance_count
# from eval_mesh import evaluate_pcs
import trimesh
import open3d as o3d
import json

exp_root = "../experiment/car_outs"
gt_root = "../data/shapenet_car/data_visualize"
score_dict = {}
score_dict_meshudf = {}
score_dict_inv = {}
score_dict_meshudf_inv = {}
score_dict_capudf = {}
bad_results = []

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


def batch_evaluate(name):
    if not os.path.exists(os.path.join(gt_root,"{}_gt_sample.ply".format(name))):
        print("not train {}".format(name))
        return
    gt_mesh = o3d.io.read_triangle_mesh(os.path.join(gt_root, "{}_gt_sample.ply".format(name)))
    # if not os.path.exists(os.path.join(exp_root,name,"mesh","{}.ply".format(name))):
    #     print("not cut result {}".format(name))
    #     return
    result_path = os.path.join(exp_root, name, "mesh", "{}.ply".format(name))
    if os.path.exists(result_path):
        our_result = o3d.io.read_triangle_mesh(result_path)
    else:
        print(name)
        bad_results.append(name)
        return
    # if isinstance(gt_mesh, trimesh.PointCloud):
    #     gt_pc = gt_mesh.vertices
    # else:
    #     gt_pc = gt_mesh.sample(100000)
    pred_meshudf = o3d.io.read_triangle_mesh(os.path.join(exp_root, name, "mesh", "{}-meshudf.ply".format(name)))
    pred_capudf = o3d.io.read_triangle_mesh(os.path.join(exp_root, name, "mesh", "0_mesh.obj"))
    # pred_pc = our_result.sample(100000)
    re = distance_count(gt_mesh, our_result, 100000)
    score_dict[name] = re
    re2 = distance_count(gt_mesh, pred_meshudf, 100000)
    score_dict_meshudf[name] = re2
    re3 = distance_count(gt_mesh, pred_capudf, 100000)
    score_dict_capudf[name] = re3
    print("***************")
    print(name)
    print(re * 10000)
    print(re2 * 10000)
    print(re3 * 10000)
    print("***************")
    return



if __name__ == "__main__":
    for p in os.listdir(exp_root):
        batch_evaluate(p)
    _sum = 0
    for d in score_dict.values():
        _sum += d['chamfer-L2']*10000
    print(_sum/len(score_dict.keys()))
    _sum = 0
    for d in score_dict_meshudf.values():
        _sum += d['chamfer-L2']*10000
    print(_sum/len(score_dict_meshudf.keys()))
    _sum = 0
    for d in score_dict_inv.values():
        _sum += d['chamfer-L2']*10000
    print(_sum/len(score_dict.keys()))
    _sum = 0
    for d in score_dict_meshudf_inv.values():
        _sum += d['chamfer-L2']*10000
    print(_sum/len(score_dict_meshudf.keys()))
    _sum = 0
    # for d in score_dict_capudf.values():
    #     _sum += d['chamfer-L2']*10000
    # print(_sum/len(score_dict_capudf.keys()))
    with open("recording_car.json", "w") as f:
        json.dump(score_dict,f,indent=4)
    with open("recording_meshudf_car.json", "w") as f:
        json.dump(score_dict_meshudf,f,indent=4)
    # with open("recording_capudf.json", "w") as f:
    #     json.dump(score_dict_capudf,f,indent=4)
    with open("recording_bad_car.json", "w") as f:
        json.dump(bad_results,f,indent=4)
