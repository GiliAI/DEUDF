import numpy as  np
import mcubes
import trimesh


from tools.logger import print_log
from skimage import measure
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

def surface_extraction(ndf, grad, out_path, iter_step, b_max, b_min, resolution):
    v_all = []
    t_all = []
    threshold = 0.005   # accelerate extraction
    v_num = 0
    for i in range(resolution-1):
        for j in range(resolution-1):
            for k in range(resolution-1):
                ndf_loc = ndf[i:i+2]
                ndf_loc = ndf_loc[:,j:j+2,:]
                ndf_loc = ndf_loc[:,:,k:k+2]
                if np.min(ndf_loc) > threshold:
                    continue
                grad_loc = grad[i:i+2]
                grad_loc = grad_loc[:,j:j+2,:]
                grad_loc = grad_loc[:,:,k:k+2]

                res = np.ones((2,2,2))
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            if np.dot(grad_loc[0][0][0], grad_loc[ii][jj][kk]) < 0:
                                res[ii][jj][kk] = -ndf_loc[ii][jj][kk]
                            else:
                                res[ii][jj][kk] = ndf_loc[ii][jj][kk]

                if res.min()<0:
                    vertices, triangles = mcubes.marching_cubes(
                        res, 0.0)
                    # print(vertices)
                    # vertices -= 1.5
                    # vertices /= 128
                    vertices[:,0] += i #/ resolution
                    vertices[:,1] += j #/ resolution
                    vertices[:,2] += k #/ resolution
                    triangles += v_num
                    # vertices =
                    # vertices[:,1] /= 3  # TODO
                    v_all.append(vertices)
                    t_all.append(triangles)

                    v_num += vertices.shape[0]
                    # print(v_num)

    v_all = np.concatenate(v_all)
    t_all = np.concatenate(t_all)
    # Create mesh
    v_all = v_all / (resolution - 1.0) * (b_max - b_min)[None, :] + b_min[None, :]

    mesh = trimesh.Trimesh(v_all, t_all, process=True)

    return mesh


def threshold_MC(ndf, threshold, resolution,bound_min=None,bound_max=None):
    try:
        vertices, triangles,_,_ = measure.marching_cubes(
                            ndf, threshold,spacing=(2/resolution,2/resolution,2/resolution))
        vertices -= 1
        # t = vertices[:,1].copy()
        # vertices[:,1] = vertices[:,2]
        # vertices[:, 2] = -t
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
    except ValueError:
        print("threshold too high")
        mesh = None

    if bound_min is not None:
        bound_min = bound_min.cpu().numpy()
        bound_max = bound_max.cpu().numpy()
        mesh.apply_scale((bound_max-bound_min)/2)
        mesh.apply_translation((bound_min+bound_max)/2)
    return mesh





