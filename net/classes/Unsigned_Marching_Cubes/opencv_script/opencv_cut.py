import cv2
import os
import matplotlib.pyplot as plt


root = "D:\\images"
gt = "gt"
tag = "meshudf"
out = "out"

out_root = os.path.join(root,out)
gt_root = os.path.join(root, gt)
tag_root = os.path.join(root,tag)
pad = 20
os.makedirs(root,exist_ok=True)
for img_name in os.listdir(gt_root):
    print(img_name)
    if img_name.startswith("0"):
        continue
    tag_name = img_name[:-6] + tag + ".png"
    img_gt = cv2.imread(os.path.join(gt_root,img_name),cv2.IMREAD_UNCHANGED)
    g_img = img_gt[:,:,-1].copy()
    img_tag = cv2.imread(os.path.join(tag_root,tag_name),cv2.IMREAD_UNCHANGED)
    g_tag = img_tag[:,:,-1].copy()
    g_img[g_img<20] = 0
    g_img[g_img>=20] = 1
    g_tag[g_tag<20] = 0
    g_tag[g_tag>=20] = 1
    x,y,w,h=cv2.boundingRect(g_img)
    xt,yt,wt,ht = cv2.boundingRect(g_tag)
    offset_x = 0
    offset_y = 0
    print((x,y,w,h))
    img_gt = img_gt[max(y-pad, 0):min(y+h+pad, img_gt.shape[0]),max(0,x-pad):min(x+w+pad,img_gt.shape[1])]
    img_tag = img_tag[max(yt+offset_y-pad, 0):min(yt+offset_y+h+pad, img_tag.shape[0]),max(0,xt+offset_x-pad):min(xt+offset_x+w+pad,img_tag.shape[1])]
    # img_gt = cv2.resize(img_gt, (1920,int(1080*(1920/w)) ))
    # img_tag = cv2.resize(img_tag, (1920,int(1080 * (1920 / w))))
    cv2.imwrite(os.path.join(out_root,img_name), img_gt)
    cv2.imwrite(os.path.join(out_root, tag_name), img_tag)

