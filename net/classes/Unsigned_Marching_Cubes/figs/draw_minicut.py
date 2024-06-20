import trimesh
import numpy as np


v = np.array([[-1,1,1],
              [1,1,1],
              [1,-1,1],
              [-1,-1,1],
              [-1, 1, -1],
              [1, 1, -1],
              [1, -1, -1],
              [-1, -1, -1]
              ])
faces = np.array([[0,1,4],
                  [1,4,5],
                  [1,2,5],
                  [2,5,6],
                  [2,3,6],
                  [3,7,6],
                  [0,3,4],
                  [3,4,7],
                  [4,7,6],
                  [4,6,5]])
faces = np.vstack((faces,faces))
mesh = trimesh.Trimesh(v,faces)
for i in range(5):
    mesh = mesh.subdivide()
mesh.export("result.ply")