import os
import json

paths = os.listdir("../data/shapenet_car/input")
paths = [p[:-4] for p in paths]
n = len(paths)
vv = 4
for i in range(vv-1):
    sub_paths = paths[i*(n//vv):(i+1)*(n//vv)]
    with open("../batch_file/cloth_{}-{}.json".format(i+1,vv), "w") as f:
        json.dump(sub_paths,f, indent=4)
with open("../batch_file/cloth_{}-{}.json".format(vv,vv), "w") as f:
    sub_paths = paths[(vv-1)*(n//vv):]
    json.dump(sub_paths,f, indent=4)