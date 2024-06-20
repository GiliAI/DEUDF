import open3d as o3d
import numpy as np
import matplotlib

pcd = o3d.io.read_point_cloud("../dense_point_cloud.ply")
pcd.paint_uniform_color([0.5, 0.5, 0.5])
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
max_nei = 0
color_map = matplotlib.colormaps["jet"]
color = np.zeros(len(pcd.points))

for i in range(len(pcd.points)):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.005)
    color[i] = len(idx)
    if len(idx) > max_nei:
        max_nei = len(idx)
        print(max_nei)
color = color/max_nei
color = 2/(1 + np.exp(-10*color))-1


color = color_map(color)[:,:3]
# for c in color:
#     print(c)
np.asarray(pcd.colors)[:] = color
o3d.io.write_point_cloud("color_pc.ply", pcd)
