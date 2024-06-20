import numpy as np
import trimesh
import os
import random
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from collections import defaultdict, deque
import itertools
import networkx as nx
import matplotlib
# import open3d as o3d
# random.seed(4)

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
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, process=False)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def compute_neighbor(face_neighbor, idx, size=2000):
    wait_list = face_neighbor[idx]
    neighbor_list = set(wait_list)
    while len(neighbor_list) < size:
        k_point = wait_list[0]
        wait_list = wait_list[1:]
        new_list = set(face_neighbor[k_point])
        new_list -= neighbor_list
        new_list -= set(wait_list)
        wait_list.extend(list(new_list))
        neighbor_list = neighbor_list.union(new_list)
    # print(len(wait_list))
    neighbor_list.union(set(wait_list))
    neighbor_list = set(neighbor_list)
    neighbor_list.remove(idx)
    return neighbor_list


def get_graph(face_neighbor, seed, seed_group_idxs, sink, sink_group_idxs):
    # （1）data表示数据，为[1, 2, 3, 4, 5, 6]
    # （2）shape表示矩阵的形状
    # （3）indices表示对应data中的数据，在压缩后矩阵中各行的下标，如：数据1在某行的0位置处，数据2在某行的2位置处，数据6在某行的2位置处。
    # （4）indptr表示压缩后矩阵中每一行所拥有数据的个数，如：[0 2 3 6]
    # 表示从第0行开始数据的个数，0 表示默认起始点
    # 0之后有几个数字就表示有几行，第一个数字2表示第一行有2 - 0 = 2 个数字，因而数字1，2
    # 都第0行，第二行有3 - 2 = 1 个数字，因而数字3在第1行，以此类推。
    neighbors = face_neighbor
    # data 全1，数量与neighbors数目相同

    # reform neighbors to seed based group gather
    new_neighbors = [ neighbors[seed], neighbors[sink]]
    neighbor_idx_map = {}
    neighbor_idx_map[seed] = 0
    neighbor_idx_map[sink] = 1
    inverse_map = {}
    inverse_map[0] = seed
    inverse_map[1] = sink
    for i in range(len(neighbors)):
        if i == seed or i == sink:
            continue
        elif i in seed_group_idxs:
            new_neighbors[0].extend(neighbors[i])
            neighbor_idx_map[i] = 0
        elif i in sink_group_idxs:
            new_neighbors[1].extend(neighbors[i])
            neighbor_idx_map[i] = 1
        else:
            new_neighbors.append(neighbors[i])
            neighbor_idx_map[i] = len(new_neighbors) - 1
            inverse_map[len(new_neighbors) - 1] = i
    for i in range(len(new_neighbors)):
        new_neighbors[i] = list(set([neighbor_idx_map[idx] for idx in new_neighbors[i]]))

    # stack neighbors to 1D arrays
    new_neighbors[0].remove(0)
    new_neighbors[1].remove(1)
    indices = np.concatenate(new_neighbors)
    indptr = [0]
    for i in range(len(new_neighbors)):
        indptr.append(indptr[-1] + len(new_neighbors[i]))
    data = np.concatenate([[1] * len(n)
                               for n in new_neighbors])
    matrix = csr_matrix((data, indices, indptr), shape=(len(new_neighbors), len(new_neighbors)))



    return matrix, inverse_map


def facetize(edges,v_num):
    """turn a set of edges into a set of consistently numbered faces"""

    # # build lookups for vertices
    # adjacent_vertices = defaultdict(set)
    # for a, b in edges:
    #     adjacent_vertices[a] |= {b}
    #     adjacent_vertices[b] |= {a}
    #
    # orderless_faces = set()
    # adjacent_faces = defaultdict(set)
    #
    # for a, b in edges:
    #     # create faces initially with increasing vertex numbers
    #     f1, f2 = (
    #         tuple(sorted([a, b, c]))
    #         for c in adjacent_vertices[a] & adjacent_vertices[b]
    #     )
    #
    #     orderless_faces |= {f1, f2}
    #     adjacent_faces[f1] |= {f2}
    #     adjacent_faces[f2] |= {f1}
    #
    #
    # def conflict(f1, f2):
    #     """returns true if the order of two faces conflict with one another"""
    #     return any(
    #         e1 == e2
    #         for e1, e2 in itertools.product(
    #             (f1[0:2], f1[1:3], f1[2:3] + f1[0:1]),
    #             (f2[0:2], f2[1:3], f2[2:3] + f2[0:1])
    #         )
    #     )
    #
    # # state for BFS
    # processed = set()
    # to_visit = deque()
    #
    # # result of BFS
    # needs_flip = {}
    #
    # # define the first face as requiring no flip
    # first = next(orderless_faces)
    # needs_flip[first] = False
    # to_visit.append(first)
    #
    # while to_visit:
    #     face = to_visit.popleft()
    #     for next_face in adjacent_faces[face]:
    #         if next_face not in processed:
    #             processed.add(next_face)
    #             to_visit.append(next_face)
    #             if conflict(next_face, face):
    #                 needs_flip[next_face] = not needs_flip[face]
    #             else:
    #                 needs_flip[next_face] = needs_flip[face]
    #
    # return [f[::-1] if needs_flip[f] else f for f in orderless_faces]

    class vertex(object):
        def __init__(self, ID):
            self.ID = ID
            self.connected = set()

        def connect(self, cVertex):
            self.connected.add(cVertex.ID)

    vertex_list = [vertex(ID) for ID in range(v_num)]
    face_list = set()
    edge_list = set()
    edges.sort(key=lambda tup: tup[0] + tup[1] / 100000000000.0)
    for (a, b) in edges:
        vertex_list[a].connect(vertex_list[b])
        vertex_list[b].connect(vertex_list[a])
        common = vertex_list[a].connected & vertex_list[b].connected
        if (common):
            print(1111)
            for x in common:
                if not set([(x, a), (a, b), (b, x)]) & edge_list:
                    face_list.add((x, a, b))
                    edge_list.update([(x, a), (a, b), (b, x)])

                elif not set([(a, x), (x, b), (b, a)]) & edge_list:
                    face_list.add((a, x, b))
                    edge_list.update([(a, x), (x, b), (b, a)])

    return face_list


def graph2mesh(mesh, graph, inverse_map):
    # convert graph to neighbor list
    data = graph.data
    indices = graph.indices
    # 0 is 0, skip
    indptr = graph.indptr
    i = 0
    neighbor_list = []
    for i in range(len(indptr)-1):
        new_node = indices[indptr[i]:indptr[i+1]]
        # new_node = new_node[data[indptr[i]:indptr[i+1]]!=0]
        neighbor_list.append(new_node)

    # convert neighbor_list to edge
    edge_list = []
    for i, new_node in enumerate(neighbor_list):
        new_edges = []
        for p in new_node:
            if p > i:
                new_edges.append((inverse_map[i],inverse_map[p]))
        edge_list.extend(new_edges)
    faces = facetize(edge_list, len(mesh.vertices))
    faces = np.array(list(faces))
    new_mesh = trimesh.base.Trimesh(vertices=mesh.vertices,faces=faces,process=False)
    return new_mesh




def cut_mesh(root):
    mesh_name = os.path.join(root,"mesh.ply")
    mesh = as_mesh(trimesh.load_mesh(mesh_name, process=False))
    mesh_name = os.path.join(root,"mesh_MC.ply")
    origin_mesh = as_mesh(trimesh.load_mesh(mesh_name, process=False))
    points = np.sum(np.array(mesh.triangles),axis=1)
    ptree = cKDTree(points)
    face_neighbor = []
    for i in range(len(points)):
        face_neighbor.append([])
    for d in mesh.face_adjacency:
        face_neighbor[d[0]].append(d[1])
        face_neighbor[d[1]].append(d[0])
    flag = True
    count = 0
    while flag:
        seed = np.random.choice(points.shape[0], 1, replace=False)[0]
        df, near_idx = ptree.query(points[seed],10)
        seed_neighbor = compute_neighbor(face_neighbor, seed, int(len(points)/20))
        seed_neighbor.add(seed)
        for idx in near_idx:
            if idx not in seed_neighbor:
                sink_neighbor = compute_neighbor(face_neighbor,idx, int(len(points)/20))
                sink_neighbor.add(idx)
                if len(sink_neighbor.intersection(seed_neighbor))>0:
                    print("seed sink overlap, might caused by too large region")
                    continue
                weight = mesh.face_adjacency_angles[:, np.newaxis].copy()
                weight = weight.max() - weight
                weight = np.exp(4*weight)
                # weight = np.cos(weight) + 1.1
                # weight = weight*10
                # weight *=20
                # weight = weight.max() - weight
                # weight = weight.astype(int)
                edges = np.concatenate((mesh.face_adjacency, weight), axis=1)
                new_idx= []
                for i in range(edges.shape[0]):
                    # edge  in seed range
                    if edges[i][0] in seed_neighbor and edges[i][1] in seed_neighbor:
                        continue
                    # edge in sink range
                    elif edges[i][0] in sink_neighbor and edges[i][1] in sink_neighbor:
                        continue
                    # start from seed range
                    elif edges[i][0] in seed_neighbor:
                        edges[i][0] = seed
                    elif edges[i][0] in sink_neighbor:
                        edges[i][0] = idx
                    elif edges[i][1] in seed_neighbor:
                        edges[i][1] = seed
                    elif edges[i][1] in sink_neighbor:
                        edges[i][1] = idx
                    new_idx.append(i)
                edges = edges[new_idx]
                edges[:,2] = edges[:,2] + 1
                inverse_edges = np.zeros_like(edges)
                inverse_edges[:,0] = edges[:,1]
                inverse_edges[:,1] = edges[:,0]
                inverse_edges[:,2] = edges[:,2]
                all_edges = np.concatenate((edges,inverse_edges),axis=0)
                visual_color = np.zeros((points.shape[0]))
                color_count = np.zeros_like(visual_color)
                color_map = matplotlib.colormaps["jet"]
                for i in range(len(mesh.face_adjacency)):
                    visual_color[mesh.face_adjacency[i][0]] += mesh.face_adjacency_angles[i]
                    visual_color[mesh.face_adjacency[i][1]] += mesh.face_adjacency_angles[i]
                    # color_count[mesh.face_adjacency[i][0]] += 1
                    # color_count[mesh.face_adjacency[i][1]] += 1
                visual_color = (visual_color / 3 / np.pi) * 2000 + 30
                visual_color[visual_color > 255] = 255
                visual_color = visual_color.astype(int)
                visual_color = color_map(visual_color)
                vertex_color = np.zeros((mesh.vertices.shape[0], 3))
                # print(vertex_color.shape)
                for i, d in enumerate(mesh.vertex_faces):
                    index = d[d != -1]
                    # print(np.average(visual_color[index],axis=0))
                    vertex_color[i] = np.average(visual_color[index], axis=0)[:3]
                # visual_color[list(curve_set)] = (0.8,0.0,0.8,1)
                visual_mesh = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_color,
                                                   process=False)
                visual_mesh.export("postprocess/cut_result/visual.ply")


        print("not sink not in seed neighbor")






if __name__ == "__main__":
    root = "postprocess/wait_for_cut"
    import time
    s = time.time()
    cut_mesh(root)
    e = time.time()
    print(e-s)
