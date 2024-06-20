import torch
import numpy as np
from network.sirenfeatureudf import PointNetfeat

if __name__=='__main__':
    featureEncoder = PointNetfeat(512,False)
    x=torch.rand(4096,128,3)
    fea1,_,_=featureEncoder(x.transpose(2,1))
    # y=np.random.choice(224,replace=False)
    x2=x[:,torch.randperm(128),:]
    fea2,_,_ = featureEncoder(x2.transpose(2,1))

    # rotate 180 
    rotate = np.array([1,1,-1]).reshape(1,1,3)
    rotate = np.repeat(rotate,128,axis=1)
    rotate = np.repeat(rotate,4096,axis=0)
    x3=x*rotate
    fea3,_,_ = featureEncoder(x3.transpose(2,1))

    x4=x+0.05
    fea4,_,_ = featureEncoder(x4.transpose(2,1))
    print(x)
    print(fea1)
    print(x2)
    print(fea2)
    print(x3)
    print(fea3)
