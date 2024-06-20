import trimesh
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


model_name = "wave2"
duration=16
fig, ax = plt.subplots(1,figsize=(10,6))
def make_frame(t):
    ax.clear()
    mesh = trimesh.load_mesh("../experiment/outs_batch/{}/mesh/{}_{}_Optimize_0.005.ply".format(model_name,model_name,int(t*10)))
    lines = trimesh.intersections.mesh_plane(mesh, (0, 1, 0), (0, 0, 0))
    for l in lines:
        ax.plot([l[0, 0], l[1, 0]], [l[0, 2], l[1, 2]], color="red")
    ax.set_ylim(-0.5, 0.5)
    return mplfig_to_npimage(fig)

animation =mpy.VideoClip(make_frame, duration=duration)
animation.write_videofile("m.mp4", fps=25)
