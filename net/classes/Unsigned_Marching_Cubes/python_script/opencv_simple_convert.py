import cv2
import os
import matplotlib.pyplot as plt


root = "D:\\images\\compare_new"
pad=20
out_root = root+"_converted"
os.makedirs(out_root,exist_ok=True)
#444
# start_x = 900
# start_y = 20
# crop_x = 400
# crop_y = int(crop_x*1.3)
#252
# start_x = 400
# start_y = 500
# crop_x = 250
# crop_y = int(crop_x*1.3)
#box
# start_x = 50
# start_y = 70
# crop_x = 300
# crop_y = int(crop_x*1.3)
#dc
# start_x = 80
# start_y = 250
# crop_x = 200
# crop_y = int(crop_x*1.3)
#mobius2
# start_x = 0
# start_y = 500
# crop_x = 200
# crop_y = int(crop_x*1.3)
# fgdj
# start_x = 200
# start_y = 150
# crop_x = 200
# crop_y = int(crop_x*1.3)
#T
# start_x = 300
# start_y = 500
# crop_x = 300
# crop_y = int(crop_x*1.3)
#sub
# start_x = 0
# start_y = 70
# crop_x = 300
# crop_y = int(crop_x*1.3)
# bunny
# start_x = 45
# start_y = 40
# crop_x = 250
# crop_y = int(crop_x*1.3)
#bed
# start_x = 200
# start_y = 60
# crop_x = 300
# crop_y = int(crop_x*1.3)
#07136
# start_x = 120
# start_y = 1850
# crop_x = 350
# crop_y = int(crop_x*1.3)

# start_x = 220
# start_y = 150
# crop_x = 200
# crop_y = int(crop_x*1.5)
start_x = 80
start_y = 250
crop_x = 200
crop_y = int(crop_x*1.3)
for img_name in os.listdir(root):
    print(img_name)
    img_gt = cv2.imread(os.path.join(root,img_name),cv2.IMREAD_UNCHANGED)
    g_img = img_gt[:,:,-1].copy()
    g_img[g_img<20] = 0
    g_img[g_img>=20] = 1
    x,y,w,h=cv2.boundingRect(g_img)
    # print((x,y,w,h))
    # # img_gt[g_img == 0] = (255, 255, 255, 0)
    #

    img_gt = img_gt[max(y-pad, 0):min(y+h+pad, img_gt.shape[0]),max(0,x-pad):min(x+w+pad,img_gt.shape[1])]
    # img_gt = img_gt[:,100:-500]
    # img_gt = img_gt[100:]
    # img_crop = img_gt[start_x:start_x+crop_x, start_y:start_y+crop_y].copy()
    # if img_gt.shape[2]==4:
    #     img_gt[img_gt[:,:,3]==0] = (255,255,255,255)
    img_gt = cv2.resize(img_gt, (500, int(img_gt.shape[0] * 500 / img_gt.shape[1])))
    cv2.imwrite(os.path.join(out_root,img_name[:-4]+".png"), img_gt)
    # cv2.imwrite(os.path.join(out_root, img_name[:-4] + "_crop.png"), img_crop)
