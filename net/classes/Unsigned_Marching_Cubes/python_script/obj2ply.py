import trimesh
import os
import shutil


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


root = "../experiment/outs_car_capudf"
out_path = "../experiment/outs_car_capudf"
for path in os.listdir(root):
    if path == "checkpoints":
        continue
    # mesh = trimesh.load_mesh()
    # mesh.export()
    if os.path.exists(os.path.join(root, path, "mesh/60000_mesh.obj")):
        mesh = as_mesh(trimesh.load_mesh(os.path.join(root, path, "mesh/60000_mesh.obj")))
        mesh.process()
        mesh.export(os.path.join(root, path, "mesh/60000_mesh.ply"))
