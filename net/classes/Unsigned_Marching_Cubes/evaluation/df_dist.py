
import numpy as np
import open3d as o3d
import open3d.geometry
import pymeshlab
import trimesh

ms = pymeshlab.MeshSet()



def distance_count(gt, pred, number_of_points):
    if np.asarray(gt.triangles).shape[0] == 0:
        pc1 = open3d.geometry.PointCloud(gt.vertices)
    else:
        pc1 = gt.sample_points_uniformly(number_of_points=number_of_points)
    if np.asarray(pred.triangles).shape[0] == 0:
        pc2 = open3d.geometry.PointCloud(pred.vertices)
    else:
        pc2 = pred.sample_points_uniformly(number_of_points=number_of_points)
    re1 = np.asarray(pc1.compute_point_cloud_distance(pc2)).mean()
    re2 = np.asarray(pc2.compute_point_cloud_distance(pc1)).mean()
    return [re1,re2,(re1+re2)/2]

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


def batch_test():
    gt_file_format = "D:\\render_data\\all_data_for_render\\high-res-sdf\\{}-gt.ply"
    pred_file_format = "D:\\render_data\\all_data_for_render\\high-res-sdf\\{}-idf-{}.ply"
    number_of_points = 10000000
    class_to_test = ["dragon", "dc", "Armadillo"]
    # method = [ "ndc","capudf", "meshudf","ours","capudf512","meshudf512", "ours512"]
    method = ["ndc" ,"mc256","mc512", "mc1024","ours","ours512","ours1024"]
    for c in class_to_test:
        print(c)

        gt_mesh = o3d.io.read_triangle_mesh(gt_file_format.format(c))
        print(calculate_topology_measure(gt_file_format.format(c)))
        # print("gt-gt")
        # # print(distance_count(pc, gt_mesh))
        # result = distance_count(gt_mesh, gt_mesh, number_of_points)
        # print("chamferl1:%.3f" % (result[2] * 1000))
        for m in method:
            pred_mesh = o3d.io.read_triangle_mesh(pred_file_format.format(c, m))
            # pc.export("gt.ply")
            if m == "mc256" or m == "ndc":
                print("trans")
                vertices = np.asarray(pred_mesh.vertices)
                vertices += 1 / 256
                pred_mesh.scale((256/255),center=(0,0,0))
                # pred_mesh.vertices = vertices
            elif m.startswith("mc"):
                print("trans")
                vertices = np.asarray(pred_mesh.vertices)
                vertices += 1 / int(m[2:])
                pred_mesh.scale((int(m[2:])/(int(m[2:])-1)),center=(0,0,0))
            print("gt-{}".format(m))
            print(calculate_topology_measure(pred_file_format.format(c, m)))
            re = distance_count(gt_mesh, pred_mesh, number_of_points)
            print("chamferl1:%.3f" % (re[2] * 1000))
            print("non-manifold edge {}%".format(100*len(pred_mesh.get_non_manifold_edges())/np.asarray(pred_mesh.vertices).shape[0]))
            print("non-manifold vertices {}%".format(100*len(pred_mesh.get_non_manifold_vertices())/np.asarray(pred_mesh.vertices).shape[0]))


def fig9_test():
    gt_file_format = "D:\\render_data\\all_data_for_render\\compare_mesh\\{}.ply"
    pred_file_format = "D:\\render_data\\all_data_for_render\\compare_mesh\\{}_{}.ply"
    number_of_points = 100000
    class_to_test = ["car1"]
    # method = ["capudf-ours", "meshudf-ours", "ours"]
    method = ["meshudf_ndf", "capudf_ndf", "ours_ndf",
        "meshudf_capudf","capudf_capudf",  "ours_capudf",
         "meshudf_ours","capudf_ours", "ours_ours"]
    for c in class_to_test:
        print(c)

        gt_mesh = o3d.io.read_triangle_mesh(gt_file_format.format(c))
        print("gt-gt")
        # print(distance_count(pc, gt_mesh))
        result = distance_count(gt_mesh, gt_mesh, number_of_points)
        print(calculate_topology_measure(gt_file_format.format(c)))
        print("chamferl1:%.3f" % (result[2] * 1000))
        for m in method:
            pred_mesh = o3d.io.read_triangle_mesh(pred_file_format.format(c, m))

            # pc.export("gt.ply")
            if m == "mc" or m == "ndc":
                print("trans")
                vertices = np.asarray(pred_mesh.vertices)
                vertices += 1 / 256
                # pred_mesh.vertices = vertices
            print("gt-{}".format(m))
            print(calculate_topology_measure(pred_file_format.format(c, m)))
            re = distance_count(gt_mesh, pred_mesh, number_of_points)
            print("chamferl1:%.3f" % (re[2] * 1000))
            print("non-manifold edge {}%".format(100*len(pred_mesh.get_non_manifold_edges())/np.asarray(pred_mesh.vertices).shape[0]))
            print("non-manifold vertices {}%".format(100*len(pred_mesh.get_non_manifold_vertices())/np.asarray(pred_mesh.vertices).shape[0]))


def single_test():
    # pre_path = "D://render_data//all_data_for_render//water_tight//dc.ply"
    gt_mesh = o3d.io.read_triangle_mesh("D://render_data//all_data_for_render//open_model_gt//564.ply")
    pre_path = "D:\\render_data\\all_data_for_render\\open_model\\564_ours.ply"
    # gt_mesh = o3d.io.read_triangle_mesh("D:\\Projects\\CXH\\Unsigned_Marching_Cubes\\postprocess\\norm\\564.ply")
    pred1 = o3d.io.read_triangle_mesh(pre_path)
    print(distance_count(gt_mesh,pred1,100000))
    print(calculate_topology_measure(pre_path))
    print("non-manifold vertices {}%".format(
        100 * len(pred1.get_non_manifold_vertices()) / np.asarray(pred1.vertices).shape[0]))
    # print("non-manifold edges {}%".format(
    #     100 * len(pred1.get_non_manifold_edges()) / trimesh.load_mesh(pre_path).edges.shape[0]))
    # print(len(pred1.cluster_connected_triangles()[1]))


if __name__ == "__main__":
    single_test()

