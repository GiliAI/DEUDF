import os

import cv2
import numpy as np
import glob

for name in os.listdir("D:\\blender_outs\\film"):
    img_array = []
    names = []
    for filename in glob.glob('D:\\blender_outs\\film\\{}\\*.png'.format(name)):
        names.append(filename)
    names.sort()
    for filename in names:
        img = cv2.imread(filename)
        # img[img[:,:,3]==0] = (255,255,255,255)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('{}.avi'.format(name), cv2.VideoWriter_fourcc(*"XVID"), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()