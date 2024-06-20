import json
import pandas

sum=0
with open("recording_capudf.json", "r") as f:
    with open("recording_capudf.csv", "w") as f_out:
        data = json.load(f)
        for d in data.keys():
            f_out.write(d)
            for i in range(len(data[d])):
                f_out.write(","+str(data[d][i]))
            f_out.write("\n")
            # sum+=data[d][2]
print(sum)


