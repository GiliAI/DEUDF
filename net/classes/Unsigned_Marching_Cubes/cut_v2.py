import numpy as np
import trimesh
import os
import time
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from collections import defaultdict, deque
import itertools
import maxflow
import networkx
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


def compute_neighbor(face_neighbor, idx, point_num, rate):
    size = int(point_num/rate)
    mask_wait = np.zeros(point_num, dtype=bool)
    # mask_wait[:] = False
    mask_wait[idx] = True
    mask_wait[face_neighbor[idx]] = True
    wait_list = face_neighbor[idx]
    neighbor_list = wait_list
    flag = True
    count = 0
    while len(neighbor_list) < size:
        if len(wait_list) == 0:
            return None
        k_point = wait_list.pop()
        for k in face_neighbor[k_point]:
            if not mask_wait[k]:
                wait_list.append(k)
                flag=False
                neighbor_list.append(k)

        if flag:
            count += 1
        else:
            count = 0
        if count == 20:
            return None
    # print(len(wait_list))
    # neighbor_list.union(set(wait_list))
    neighbor_list = set(neighbor_list)
    print(len(neighbor_list))
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
    new_neighbors = [neighbors[seed], neighbors[sink]]
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
        new_neighbors[i] = list(set([neighbor_idx_map[idx]
                                for idx in new_neighbors[i]]))

    # stack neighbors to 1D arrays
    new_neighbors[0].remove(0)
    new_neighbors[1].remove(1)
    indices = np.concatenate(new_neighbors)
    indptr = [0]
    for i in range(len(new_neighbors)):
        indptr.append(indptr[-1] + len(new_neighbors[i]))
    data = np.concatenate([[1] * len(n)
                           for n in new_neighbors])
    matrix = csr_matrix((data, indices, indptr), shape=(
        len(new_neighbors), len(new_neighbors)))

    return matrix, inverse_map


def facetize(edges, v_num):
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
                new_edges.append((inverse_map[i], inverse_map[p]))
        edge_list.extend(new_edges)
    faces = facetize(edge_list, len(mesh.vertices))
    faces = np.array(list(faces))
    new_mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices, faces=faces, process=False)
    return new_mesh





def cut_mesh_v2(mesh,region_rate=30):
    # cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=len(mesh.vertices)//5)
    # mask = np.zeros(len(mesh.faces), dtype=np.bool)
    # mask[np.concatenate(cc)] = True
    # mesh.update_faces(mask)
    # mesh.remove_unreferenced_vertices()
    points = np.sum(np.array(mesh.triangles), axis=1)
    ptree = cKDTree(points)
    face_neighbor = []
    out_mesh = None
    for i in range(len(points)):
        face_neighbor.append([])
    for d in mesh.face_adjacency:
        face_neighbor[d[0]].append(d[1])
        face_neighbor[d[1]].append(d[0])
    flag = True
    try_time = 0
    ite_num = 0
    while flag:
        if ite_num > 10:
            if try_time>20:
                print("not OK cut")
                break
            print("decrease region")
            ite_num=0
            region_rate +=10
            try_time +=1
        seed = np.random.choice(points.shape[0], 1, replace=False)[0]
        df, near_idx = ptree.query(points[seed], 50)
        near_idx = near_idx-1
        seed_neighbor = compute_neighbor(
            face_neighbor, seed, len(points),region_rate)
        if seed_neighbor is None:
            continue
        seed_neighbor.add(seed)
        idx_flag = False
        for idx in near_idx:
            # print(idx)
            if idx not in seed_neighbor:
                idx_flag=True
                s = time.time()
                sink_neighbor = compute_neighbor(
                    face_neighbor, idx, len(points),region_rate)
                if sink_neighbor is None:
                    continue
                sink_neighbor.add(idx)
                if len(sink_neighbor.intersection(seed_neighbor)) > 0:
                    # print("seed sink overlap, might caused by too large region")
                    break
                weight = mesh.face_adjacency_angles[:, np.newaxis].copy()
                e = time.time()
                print('st time ', e-s)
                # weight = np.cos(weight) + 1.1
                # weight = weight*10
                # weight *=20
                weight = weight.max() - weight
                weight = np.exp(4*weight)
                # weight[weight < 0.5] = 1e11
                # weight[weight < 1] = 1e8
                # weight[weight < 1.5] = 1e5
                # weight[weight < 2] = 1e3
                # weight[weight >1] = 1e7
                # weight[weight < 3] = 1e1
                # weight[weight < 4] = 0
                # weight = weight.astype(int)
                edges = np.concatenate((mesh.face_adjacency, weight), axis=1)
                new_idx = []
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
                edges[:, 2] = edges[:, 2] + 1
                # inverse_edges = np.zeros_like(edges)
                # inverse_edges[:, 0] = edges[:, 1]
                # inverse_edges[:, 1] = edges[:, 0]
                # inverse_edges[:, 2] = edges[:, 2]
                # all_edges = np.concatenate((edges, inverse_edges), axis=0)
                count = 0
                ori_g = [-1]*len(points)
                g_ori = [-1]*len(points)
                for e in edges:
                    if(e[0] == seed or e[0] == idx):
                        if(ori_g[int(e[1])] == -1):
                            ori_g[int(e[1])] = count
                            g_ori[count] = e[1]
                            count = count+1
                    elif(e[1] == seed or e[1] == idx):
                        if(ori_g[int(e[0])] == -1):
                            ori_g[int(e[0])] = count
                            g_ori[count] = e[0]
                            count = count+1
                    else:
                        if(ori_g[int(e[0])] == -1):
                            ori_g[int(e[0])] = count
                            g_ori[count] = e[0]
                            count = count+1
                        if(ori_g[int(e[1])] == -1):
                            ori_g[int(e[1])] = count
                            g_ori[count] = e[1]
                            count = count+1
                g_ori.remove(-1.0)
                g = maxflow.Graph[float]()
                nodes = g.add_nodes(len(g_ori))
                for e in edges:
                    if(e[0] == seed):
                        g.add_tedge(nodes[ori_g[int(e[1])]], e[2], 0)
                    elif(e[0] == idx):
                        g.add_tedge(nodes[ori_g[int(e[1])]], 0, e[2])
                    elif(e[1] == seed):
                        g.add_tedge(nodes[ori_g[int(e[0])]], e[2], 0)
                    elif(e[1] == idx):
                        g.add_tedge(nodes[ori_g[int(e[0])]], 0, e[2])
                    else:
                        g.add_edge(nodes[ori_g[int(e[0])]],
                                   nodes[ori_g[int(e[1])]], e[2], e[2])
                flow = g.maxflow()
                print('flow ', flow)
                seg = g.get_grid_segments(nodes)
                partition = (set(), set())
                for id in range(seg.shape[0]):
                    if g_ori[id]<0:
                        continue
                    if(seg[id]):
                        partition[1].add(g_ori[id])
                    else:
                        partition[0].add(g_ori[id])
                rate = len(partition[0])/len(partition[1])
                print(len(partition[0]), len(partition[1]))
                e = time.time()
                print('cutting time ', e-s)
                if rate > 1.2 or rate <0.8:
                    ite_num += 1
                    print("not a OK cut, rate is {}".format(rate))
                    break
                else:
                    flag = False

                seed_list = np.array(
                    list(partition[0].union(seed_neighbor)), dtype=int).tolist()
                sink_list = np.array(
                    list(partition[1].union(sink_neighbor)), dtype=int).tolist()

                out_mesh_1 = trimesh.base.Trimesh(
                        vertices=mesh.vertices, faces=mesh.faces[seed_list], process=False)
                out_mesh_2 = trimesh.base.Trimesh(
                        vertices=mesh.vertices, faces=mesh.faces[sink_list], process=False)
                out_mesh_1.remove_unreferenced_vertices()
                out_mesh_2.remove_unreferenced_vertices()
                # out_mesh_1.merge_vertices()
                # out_mesh_1.remove_duplicate_faces()
                # out_mesh_2.merge_vertices()
                # out_mesh_2.remove_duplicate_faces()
                return (out_mesh_1,out_mesh_2)
                # out_mesh.merge_vertices()
                # out_mesh.remove_duplicate_faces()
                # origin_seed_mesh = trimesh.base.Trimesh(vertices=origin_mesh.vertices,faces = mesh.faces[seed_list])
                # origin_sink_mesh = trimesh.base.Trimesh(vertices=origin_mesh.vertices,faces = mesh.faces[sink_list])


                # visual_color[seed_list] = (0.8,0,0)

                ite_num += 1
                # flag = False
                # flag = False
                break
                # base_graph, inverse_map = get_graph(mesh, seed, seed_neighbor, idx, sink_neighbor)
                # result = maximum_flow(base_graph, 0, 1)
                # if result.flow_value != 0:
                #     new_mesh = graph2mesh(mesh,result.flow, inverse_map)
                #     new_mesh.export("result.obj")
                #     print("finished")
                #     break

        if not idx_flag:
            ite_num += 1
            print("near region may too small")
        # print("no sink not in seed neighbor")
    if out_mesh is not None:
        out_mesh.remove_duplicate_faces()
    return out_mesh


if __name__ == "__main__":
    root = "postprocess/wait_for_cut"
    for path in os.listdir(root):
        s = time.time()
        mesh_name = os.path.join(root, path)
        mesh = as_mesh(trimesh.load_mesh(mesh_name, process=False))
        out = cut_mesh_v2(mesh)
        out[0].export(
            "postprocess/cut_result/{}-0_merge.ply".format(path[:-4]))
        out[1].export(
            "postprocess/cut_result/{}-1_merge.ply".format(path[:-4]))
        e = time.time()
        print('total ', e-s)
    # root = "experiment/out_cloth_batch"
    # for path in os.listdir(root):
    #     print(path)
    #     s = time.time()
    #     mesh_name = os.path.join(root, path,"mesh","{}_399_Optimize.ply".format(path))
    #     mesh = as_mesh(trimesh.load_mesh(mesh_name, process=False))
    #     out = cut_mesh_v2(mesh)
    #     if out is not None:
    #         out[0].export(
    #              os.path.join(root, path,"mesh","{}-0_merge.ply".format(path)))
    #         out[1].export(
    #              os.path.join(root, path,"mesh","{}-1_merge.ply".format(path)))
    #     e = time.time()
    #     print('total ', e - s)
