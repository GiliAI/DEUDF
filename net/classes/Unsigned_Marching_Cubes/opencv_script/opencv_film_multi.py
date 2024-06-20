import os

import cv2
import numpy as np
import glob

def frame_concate_v(imgs_series, k):
    origin_img = imgs_series[0][k]
    for i in range(len(imgs_series)-1):
        origin_img = cv2.vconcat([origin_img, imgs_series[i+1][k]])
    return origin_img

def frame_concate_h(imgs_series, k):
    origin_img = imgs_series[0][k]
    for i in range(len(imgs_series)-1):
        origin_img = cv2.hconcat([origin_img, imgs_series[i+1][k]])
    return origin_img

if __name__ == "__main__":
    resolution = 4
    paths = []
    for name in os.listdir("D:\\blender_outs\\film"):
        paths.append(name)
    paths.sort()



    sub_img_arrays = []
    for c, name in enumerate(paths):
        sub_img_array = []
        if c == 64:
            break
        for filename in glob.glob('D:\\blender_outs\\film\\{}\\*.png'.format(name)):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (resolution*width, resolution*height)
            sub_img_array.append(img)
        sub_img_arrays.append(sub_img_array)
    sub2_image_arrays = []
    for i in range(resolution):
        sub2_image_array = []
        for j in range(len(sub_img_arrays[0])):
            sub2_image_array.append(frame_concate_v(sub_img_arrays[resolution*i:resolution*i+resolution], j))
        sub2_image_arrays.append(sub2_image_array)
    img_array = []
    for i in range(resolution):
        for j in range(len(sub_img_arrays[0])):
            img_array.append(frame_concate_h(sub2_image_arrays, j))
        sub2_image_arrays.append(sub2_image_array)


    out = cv2.VideoWriter('multi.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()