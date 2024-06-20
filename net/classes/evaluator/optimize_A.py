import argparse
import torch
import numpy as np
from scipy.sparse import coo_matrix
import trimesh
from torch.nn import functional as F
import sys
import os
import tqdm
from collections import defaultdict


# import lib.workspace as ws

import sys
sys.path.append('custom_mc')
from _marching_cubes_lewiner import udf_mc_lewiner

def fourier_transform(x, L=5):
    cosines = torch.cat([torch.cos(2**l*3.1415*x) for l in range(L)], -1)
    sines = torch.cat([torch.sin(2**l*3.1415*x) for l in range(L)], -1)
    transformed_x = torch.cat((cosines,sines),-1)
    return transformed_x

def get_udf_normals_grid_slow(decoder, latent_vec, N=56, max_batch=int(2 ** 20), fourier=False):
    """
    Fills a dense N*N*N regular grid by querying the decoder network
    Inputs:
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        N: grid size
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
    Returns:
        df_values: (N,N,N) tensor representing distance field values on the grid
        vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
    """
    decoder.eval()
    ################
    # 1: setting up the empty grid
    ################
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 7)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = torch.div(overall_index, N, rounding_mode='floor') % N
    samples[:, 0] = torch.div(torch.div(overall_index, N, rounding_mode='floor'), N, rounding_mode='floor') % N
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    num_samples = N ** 3
    samples.requires_grad = False
    samples.pin_memory()
    ################
    # 2: Run forward pass to fill the grid
    ################
    head = 0
    ## FIRST: fill distance field grid without gradients
    while head < num_samples:
        # xyz coords
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].clone().cuda()
        # Create input
        if fourier:
            # xyz = ws.fourier_transform(sample_subset)
            xyz = fourier_transform(sample_subset)
        else:
            xyz = sample_subset
        batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, sample_subset.shape[0], 1)
        input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), xyz.reshape(-1, xyz.shape[-1])], dim=1)
        # Run forward pass
        with torch.no_grad():
            df = decoder(input)
        # Store df
        samples[head : min(head + max_batch, num_samples), 3] = df.squeeze(-1).detach().cpu()
        # Next iter
        head += max_batch
    #
    ## THEN: compute gradients only where needed,
    # ie. where the predicted df value is small
    max_batch = max_batch // 4
    norm_mask = samples[:, 3] < 2 * voxel_size
    norm_idx = torch.where(norm_mask)[0]
    head, num_samples = 0, norm_idx.shape[0]
    while head < num_samples:
        # Find the subset of indices to compute:
        # -> a subset of indices where normal computations are needed
        sample_subset_mask = torch.zeros_like(norm_mask)
        sample_subset_mask[norm_idx[head]: norm_idx[min(head + max_batch, num_samples) - 1] + 1] = True
        sample_subset_mask = norm_mask * sample_subset_mask
        # xyz coords
        sample_subset = samples[sample_subset_mask, 0:3].clone().cuda()
        sample_subset.requires_grad = True
        # Create input
        if fourier:
            # xyz = ws.fourier_transform(sample_subset)
            xyz = fourier_transform(sample_subset)
        else:
            xyz = sample_subset
        batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, sample_subset.shape[0], 1)
        input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), xyz.reshape(-1, xyz.shape[-1])], dim=1)
        # Run forward pass
        df = decoder(input)
        # Compute and store normalized vectors pointing towards the surface
        df.sum().backward()
        grad = sample_subset.grad.detach()
        samples[sample_subset_mask, 4:] = - F.normalize(grad, dim=1).cpu()
        # Next iter
        head += max_batch
    #
    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(N, N, N)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(N, N, N, 3)
    return df_values, vecs, samples


def get_mesh_udf_fast(data,t):
    """
    Computes a triangulated mesh from a distance field network conditioned on the latent vector
    Inputs:
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                    for a previous latent_vec, which is assumed to be close to the current one, if any
        indices: tensor representing the coordinates that need updating in the previous samples tensor (to speed
                    up iterations)
        N_MC: grid size
        fourier: are xyz coordinates encoded with fourier?
        gradient: do we need gradients?
        eps: length of the normal vectors used to derive gradients
        border_gradients: add a special case for border gradients?
        smooth_borders: do we smooth borders with a Laplacian?
    Returns:
        verts: vertices of the mesh
        faces: faces of the mesh
        mesh: trimesh object of the mesh
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
        indices: tensor representing the coordinates that need updating in the next iteration
    """
    ### 1: sample grid
    df_values = np.abs(data["df"])
    # print(df_values.min())
    # df_values = df_values-df_values.min()
    # df_values = df_values*10
    normals = -data["grad"]
    normals[df_values>t] = 0
    # np.savez("meshudf.npz", df=df_values, grad=normals)
    df_values[df_values < 0] = 0
    ### 2: run our custom MC on it
    N = df_values.shape[0]
    voxel_size = 2.0 / (N - 1)
    voxel_origin = [-1, -1, -1]
    verts, faces, _, _ = udf_mc_lewiner(df_values,
                                        normals,
                                        spacing=[voxel_size] * 3)
    verts = verts - 1  # since voxel_origin = [-1, -1, -1]
    filtered_mesh = trimesh.Trimesh(verts, faces)
    # filtered_mesh.export("mesh_origin.ply")
    ### 4: clean the mesh a bit
    # Remove NaNs, flat triangles, duplicate faces
    filtered_mesh = filtered_mesh.process(
        validate=False)  # DO NOT try to consistently align winding directions: too slow and poor results
    filtered_mesh.remove_duplicate_faces()
    filtered_mesh.remove_degenerate_faces()
    # Fill single triangle holes
    filtered_mesh.fill_holes()

    filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)
    # Re-process the mesh until it is stable:
    n_verts, n_faces, n_iter = 0, 0, 0
    while (n_verts, n_faces) != (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces)) and n_iter < 10:
        filtered_mesh_2 = filtered_mesh_2.process(validate=False)
        filtered_mesh_2.remove_duplicate_faces()
        filtered_mesh_2.remove_degenerate_faces()
        (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
        n_iter += 1
        filtered_mesh_2 = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

    filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)
    # filtered_mesh.export("mesh_1.ply")
    # Identify borders: those appearing only once
    border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

    # Build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
    neighbours = defaultdict(lambda: [])
    for (u, v) in filtered_mesh.edges_sorted[border_edges]:
        neighbours[u].append(v)
        neighbours[v].append(u)
    border_vertices = np.array(list(neighbours.keys()))

    # Build a sparse matrix for computing laplacian
    pos_i, pos_j = [], []
    for k, ns in enumerate(neighbours.values()):
        for j in ns:
            pos_i.append(k)
            pos_j.append(j)

    sparse = coo_matrix((np.ones(len(pos_i)),  # put ones
                         (pos_i, pos_j)),  # at these locations
                        shape=(len(border_vertices), len(filtered_mesh.vertices)))

    # Smoothing operation:
    lambda_ = 0.3
    for _ in range(5):
        border_neighbouring_averages = sparse @ filtered_mesh.vertices / sparse.sum(axis=1)
        laplacian = border_neighbouring_averages - filtered_mesh.vertices[border_vertices]
        filtered_mesh.vertices[border_vertices] = filtered_mesh.vertices[border_vertices] + lambda_ * laplacian
    bound_max = data["bound_max"]
    bound_min = data["bound_min"]


    filtered_mesh.apply_scale((bound_max - bound_min) / 2)
    filtered_mesh.apply_translation((bound_min + bound_max) / 2)
    return torch.tensor(filtered_mesh.vertices).float().cuda(), torch.tensor(
            filtered_mesh.faces).long().cuda(), filtered_mesh

def getMeshUDF(query_func,resolution,bound_min = [-1,-1,-1],bound_max=[1,1,1],threshold = 0.1):
    df, grad = extract_fields(query_func,resolution,bound_min,bound_max)
    if isinstance(bound_min, list):
        bound_min = np.array(bound_min, dtype=np.float32)
    if isinstance(bound_max, list):
        bound_max = np.array(bound_max, dtype=np.float32)
    data = {"df":df,"grad":grad,"bound_min":bound_min,"bound_max":bound_max}
    _, _, mesh_start = get_mesh_udf_fast(data,threshold)
    return mesh_start


def extract_fields(query_func,resolution,bound_min,bound_max):

    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    grad = np.zeros([resolution, resolution, resolution,3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                val = query_func(pts)
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = \
                    val.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                grad[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = \
                    torch.autograd.grad(val, [pts], grad_outputs=torch.ones_like(val),retain_graph=True,create_graph=True)[0]\
                        .reshape(len(xs), len(ys), len(zs),-1).detach().cpu().numpy()
    return u,grad

# def main_function(root, path, t):
#     data = np.load(os.path.join(root, path))
#     _, _, mesh_start = get_mesh_udf_fast(data,t)
#     _ = mesh_start.export(os.path.join("out", '{}.ply'.format(path[:-4])))


# if __name__ == "__main__":
#     arg_parser = argparse.ArgumentParser(
#         description="Reconstruct training shapes with a trained DeepSDF autodecoder (+latents)")
#     arg_parser.add_argument(
#         "--threshold", "-t", type=float, required=True,
#     )
#     args = arg_parser.parse_args()
#     t = args.threshold
#     root = "npz_file"
#     for path in os.listdir(root):
#         main_function(root, path, t)
