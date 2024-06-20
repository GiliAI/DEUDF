import trimesh
from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d

from scipy.sparse import coo_matrix


def laplacian_calculation(mesh, equal_weight=True):
    """
    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neightbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors
    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(np.ndarray)

    # stack neighbors to 1D arrays
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = np.ones(3)
        # the distance from verticesex to neighbors
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        # normalize group and stack into single array
        data = np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)

    return matrix

# mesh = trimesh.load("D:\\Projects\\CXH\\ndf\\experiment\\outs_batch\\Armadillo\\mesh\\Armadillo_pc.ply")
mesh = trimesh.load_mesh("experiment/out_cloth_batch/165_noise/mesh/165_noise_pc.ply")
mesh = trimesh.load("D:\\idf-main\\runs\\Displace_Siren_asian_dragon_phased_scaledTanh_yes_act_yes_baseLoss_yes\\Train\\File\\pts_mesh_HighRes_chamfer_gtToEval_normal_mesh_118907.ply")
# mesh = trimesh.load("D:\\Projects\\CXH\\ndf\\experiments\\shapenet_cars\\evaluation\\generation\\02958343\\1a4ef4a2a639f172f13d1237e1429e9e\\dense_point_cloud_7.off")
# total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
# centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
# mesh.apply_translation(-centers)
# mesh.apply_scale(1 / total_size)
# mesh.export("-2.ply")
gt_pc = np.asarray(mesh.vertices).copy()
# idx = np.random.choice(gt_pc.shape[0],1000000,replace=False)
gt_pc = gt_pc
# noise = np.random.randn(int(gt_pc.shape[0]*0.1),3)*0.001
# idx = np.random.choice(gt_pc.shape[0], int(gt_pc.shape[0]*0.1))
# gt_pc[idx] =gt_pc[idx] +noise
ptree = cKDTree(gt_pc)
init_mesh = trimesh.load("D:\\idf-main\\experiment\\outs_batch\\test\\mesh\\MC_mesh.ply")
# init_mesh = trimesh.load("D:\\Projects\\CXH\\ndf\\experiment\\outs_batch\\Armadillo\\mesh\\MC_mesh.ply")
# init_mesh = trimesh.load_mesh("D:\\Projects\\CXH\\ndf\\experiment\\outs_batch\\test\\mesh\\MC_mesh.ply")
# init_mesh = trimesh.load_mesh("experiment/out_cloth_batch/165_noise/mesh/MC_mesh.ply")
trimesh.PointCloud(gt_pc).export("-1.ply")
lap_mat = laplacian_calculation(init_mesh)
for i in range(10):
    v = init_mesh.vertices
    df, near_idx = ptree.query(v)
    v = v + 0.5*(gt_pc[near_idx]-v)
    init_mesh.vertices = v
    lap_vecs = (lap_mat*init_mesh.vertices)-init_mesh.vertices
    normal = np.asarray(init_mesh.vertex_normals).copy()
    angle = np.sum(normal*lap_vecs,axis=-1)
    normal[angle<0] = -normal[angle<0]
    sub_v = np.array(np.sum(lap_vecs*normal,axis=-1))[:,None]*normal
    mov_v = lap_vecs - sub_v
    init_mesh.vertices = init_mesh.vertices+0*mov_v
    init_mesh.export("{}.ply".format(i))

# mesh = trimesh.load_mesh("out.ply")
# o3d_mesh = o3d.io.read_triangle_mesh("out.ply")
# f = o3d_mesh.get_self_intersecting_triangles()
# face_id = np.asarray(f)
# print(face_id)
# mesh.export("{}.ply".format(i//10))
