import trimesh
import os
import numpy as np


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


root = "../postprocess/norm"
out = root
os.makedirs(out,exist_ok=True)
for path in os.listdir(root):
    if not path.endswith(".ply"):
        continue
    mesh_path = os.path.join(root, path)
    mesh = trimesh.load(mesh_path)
    # if isinstance(mesh, trimesh.PointCloud):
    #     continue
    # mesh = as_mesh(mesh)
    # v = np.array(mesh.vertices)
    # v[:,2] = -v[:,2]
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)
    # new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    mesh.export(os.path.join(out,path))