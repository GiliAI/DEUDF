import numpy as np
import trimesh
import os
from trimesh import creation, transformations

render_color = [[[[1., 0., 0.16 ,      1.,],
   [1., 0.6009539,  0., 1.,]],

  [[0.61473238 ,1., 0., 1.,],
   [0., 1., 0.14758592, 1.,]]],


 [[[0., 1., 0.9276829 , 1.,],
   [0., 0.30583973, 1., 1.,]],

  [[0.48273657, 0., 1., 1.,],
   [1., 0., 0.75,       1.,]]]]
root = "postprocess/wait_for_sep"
out = "postprocess/sep_data"
os.makedirs(out, exist_ok=True)
for path in os.listdir(root):
    mesh_name = os.path.join(root, path)
    mesh = trimesh.load_mesh(mesh_name)
    pointcloud = np.asarray(mesh.vertices)
    shape_scale = np.max(
        [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
         np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
    shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                    (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                    (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale
    mesh = trimesh.PointCloud(pointcloud)
    rot_matrix = transformations.rotation_matrix(np.pi / 3, [0, 0, 1], [0, 0, 0])
    mesh.apply_transform(rot_matrix)
    v = mesh.vertices
    v = v[v[:, 1] > mesh.bounds[0][1] + 0.1]
    v = v[v[:, 2] < mesh.bounds[1][2] - 0.1]
    pointcloud = v
    pad = 0.0
    size = 8
    X = np.linspace(-1 ,1 , size+1)
    Y = np.linspace(-1 ,1 , size+1)
    Z = np.linspace(-1 ,1 , size+1)
    c = 0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                c += 1
                spe_points = pointcloud[pointcloud[:,0]>X[i]-pad]
                spe_points = spe_points[spe_points[:, 0] < X[i+1]+pad]
                spe_points = spe_points[spe_points[:,1]>Y[j]-pad]
                spe_points = spe_points[spe_points[:, 1] < Y[j+1]+pad]
                spe_points = spe_points[spe_points[:,2]>Z[k]-pad]
                spe_points = spe_points[spe_points[:, 2] < Z[k+1]+pad]
                if spe_points.shape[0]<5:
                    print("seem not have sample at {} {} {}".format(X[i], Y[j], Z[k]))
                    continue
                p = trimesh.PointCloud(spe_points)

                p.export(os.path.join(out,"big_sence_{}.ply".format(c)))

