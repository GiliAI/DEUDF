import trimesh
import numpy as np


size = 0.4
v = np.array([[size,-size,-size],
              [size,size,-size],
              [-size,size,-size],
              [-size,-size,-size],
              [size,-size,size],
              [size,size,size],
              [-size,size,size],
              [-size,-size,size],
              ])

faces = np.array([[0,1,2],
                 [0,2,3],
                 [0,1,5],
                 [0,5,4],
                 [1,2,6],
                 [1,6,5],
                 [2,3,6],
                 [3,6,7],
                 [0,3,7],
                 [0,4,7],
                 [4,5,6],
                 [4,7,6]])
mesh = trimesh.Trimesh(vertices=v,faces=faces)
mesh.export("box_gt.ply")