import json
import matplotlib.pyplot as plt
import numpy as np
with open("recording.json", "r") as f:
    ours = json.load(f)
with open("recording_meshudf.json", "r") as f:
    meshudf = json.load(f)
with open("recording_capudf.json", "r") as f:
    capudf = json.load(f)

result_ours = []
result_meshudf = []
result_capudf = []
c1 = 0
c2 = 0
for k in ours.keys():
    if ours[k]['chamfer-L2']*1000>0.02:
        # print(k)
        continue
    result_ours.append(ours[k]['chamfer-L2']*1000)
    result_capudf.append(capudf[k]['chamfer-L2'] * 1000)
    if ours[k]['chamfer-L2']<meshudf[k]['chamfer-L2']:
        c1+=1
    else:
        c2+=1
        print(k)
    result_meshudf.append(meshudf[k]['chamfer-L2'] * 1000)
print(c1)
print(c2)
x = np.arange(len(result_ours))
plt.scatter(x,result_ours)
# plt.plot(result_capudf)
plt.scatter(x,result_meshudf)

plt.show()