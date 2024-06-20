import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

class CAPUDFNetwork(nn.Module):
    def __init__(self,
                 d_out,
                 d_hidden,
                 n_layers,
                 d_in=3,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(CAPUDFNetwork, self).__init__()
        self.size = 0.4
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

    def forward(self, inputs):

        region_xu = inputs[:,0]>=self.size
        region_xd = inputs[:,0]<=-self.size
        region_xm = torch.logical_not(torch.logical_or(region_xu, region_xd))
        region_yu = inputs[:,1]>=self.size
        region_yd = inputs[:,1]<=-self.size
        region_ym = torch.logical_not(torch.logical_or(region_yd, region_yu))
        region_zu = inputs[:,2]>=self.size
        region_zd = inputs[:,2]<=-self.size
        region_zm = torch.logical_not(torch.logical_or(region_zu, region_zd))
        res = torch.zeros(inputs.shape[0])
        idx_1 = torch.logical_and(region_xu, torch.logical_and(region_yu, region_zu))
        res[idx_1] = torch.norm(inputs[idx_1] - torch.FloatTensor((self.size,self.size,self.size)).cuda(), dim=1)

        idx_2 = torch.logical_and(region_xd, torch.logical_and(region_yu, region_zu))
        res[idx_2] = torch.norm(inputs[idx_2] - torch.FloatTensor((-self.size,self.size,self.size)).cuda(), dim=1)

        idx_3 = torch.logical_and(region_xu, torch.logical_and(region_yd, region_zu))
        res[idx_3] = torch.norm(inputs[idx_3] - torch.FloatTensor((self.size,-self.size,self.size)).cuda(), dim=1)

        idx_4 = torch.logical_and(region_xu, torch.logical_and(region_yu, region_zd))
        res[idx_4] = torch.norm(inputs[idx_4] - torch.FloatTensor((self.size, self.size, -self.size)).cuda(), dim=1)

        idx_5 = torch.logical_and(region_xd, torch.logical_and(region_yd, region_zu))
        res[idx_5] = torch.norm(inputs[idx_5] - torch.FloatTensor((-self.size, -self.size, self.size)).cuda(), dim=1)

        idx_6 = torch.logical_and(region_xd, torch.logical_and(region_yu, region_zd))
        res[idx_6] = torch.norm(inputs[idx_6] - torch.FloatTensor((-self.size, self.size, -self.size)).cuda(), dim=1)

        idx_7 = torch.logical_and(region_xu, torch.logical_and(region_yd, region_zd))
        res[idx_7] = torch.norm(inputs[idx_7] - torch.FloatTensor((self.size, -self.size, -self.size)).cuda(), dim=1)

        idx_8 = torch.logical_and(region_xd, torch.logical_and(region_yd, region_zd))
        res[idx_8] = torch.norm(inputs[idx_8] - torch.FloatTensor((-self.size, -self.size, -self.size)).cuda(), dim=1)

        #xm
        idx_9 = torch.logical_and(region_xm, torch.logical_and(region_yu, region_zu))
        vector_9 = inputs[idx_9] - torch.FloatTensor((0, self.size, self.size)).cuda()
        vector_9[:,0] = 0
        res[idx_9] = torch.norm(vector_9, dim=1)

        idx_10 = torch.logical_and(region_xm, torch.logical_and(region_yu, region_zd))
        vector_10 = inputs[idx_10] - torch.FloatTensor((0, self.size, -self.size)).cuda()
        vector_10[:,0] = 0
        res[idx_10] = torch.norm(vector_10, dim=1)

        idx_11 = torch.logical_and(region_xm, torch.logical_and(region_yd, region_zd))
        vector_11 = inputs[idx_11] - torch.FloatTensor((0, -self.size, -self.size)).cuda()
        vector_11[:,0] = 0
        res[idx_11] = torch.norm(vector_11, dim=1)

        idx_12 = torch.logical_and(region_xm, torch.logical_and(region_yd, region_zu))
        vector_12 = inputs[idx_12] - torch.FloatTensor((0, -self.size, self.size)).cuda()
        vector_12[:,0] = 0
        res[idx_12] = torch.norm(vector_12, dim=1)

        #ym
        idx_13 = torch.logical_and(region_xu, torch.logical_and(region_ym, region_zu))
        vector_13 = inputs[idx_13] - torch.FloatTensor((self.size, 0, self.size)).cuda()
        vector_13[:, 1] = 0
        res[idx_13] = torch.norm(vector_13, dim=1)

        idx_14 = torch.logical_and(region_xd, torch.logical_and(region_ym, region_zu))
        vector_14 = inputs[idx_14] - torch.FloatTensor((-self.size, 0, self.size)).cuda()
        vector_14[:, 1] = 0
        res[idx_14] = torch.norm(vector_14, dim=1)

        idx_15 = torch.logical_and(region_xu, torch.logical_and(region_ym, region_zd))
        vector_15 = inputs[idx_15] - torch.FloatTensor((self.size, 0, -self.size)).cuda()
        vector_15[:, 1] = 0
        res[idx_15] = torch.norm(vector_15, dim=1)

        idx_16 = torch.logical_and(region_xd, torch.logical_and(region_ym, region_zd))
        vector_16 = inputs[idx_16] - torch.FloatTensor((-self.size, 0, -self.size)).cuda()
        vector_16[:, 1] = 0
        res[idx_16] = torch.norm(vector_16, dim=1)

        #zm
        idx_17 = torch.logical_and(region_xu, torch.logical_and(region_yu, region_zm))
        vector_17 = inputs[idx_17] - torch.FloatTensor((self.size, self.size, 0)).cuda()
        vector_17[:, 2] = 0
        res[idx_17] = torch.norm(vector_17, dim=1)

        idx_18 = torch.logical_and(region_xd, torch.logical_and(region_yu, region_zm))
        vector_18 = inputs[idx_18] - torch.FloatTensor((-self.size, self.size, 0)).cuda()
        vector_18[:, 2] = 0
        res[idx_18] = torch.norm(vector_18, dim=1)

        idx_19 = torch.logical_and(region_xu, torch.logical_and(region_yd, region_zm))
        vector_19 = inputs[idx_19] - torch.FloatTensor((self.size, -self.size, 0)).cuda()
        vector_19[:, 2] = 0
        res[idx_19] = torch.norm(vector_19, dim=1)

        idx_20 = torch.logical_and(region_xd, torch.logical_and(region_yd, region_zm))
        vector_20 = inputs[idx_20] - torch.FloatTensor((-self.size, -self.size, 0)).cuda()
        vector_20[:, 2] = 0
        res[idx_20] = torch.norm(vector_20, dim=1)

        # not xm
        idx_21 = torch.logical_and(region_xu, torch.logical_and(region_ym, region_zm))
        vector_21 = inputs[idx_21] - torch.FloatTensor((self.size, 0, 0)).cuda()
        vector_21[:, 1] = 0
        vector_21[:, 2] = 0
        res[idx_21] = torch.norm(vector_21, dim=1)

        idx_22 = torch.logical_and(region_xd, torch.logical_and(region_ym, region_zm))
        vector_22 = inputs[idx_22] - torch.FloatTensor((-self.size, 0, 0)).cuda()
        vector_22[:, 1] = 0
        vector_22[:, 2] = 0
        res[idx_22] = torch.norm(vector_22, dim=1)

        # not ym
        idx_23 = torch.logical_and(region_xm, torch.logical_and(region_yu, region_zm))
        vector_23 = inputs[idx_23] - torch.FloatTensor((0, self.size, 0)).cuda()
        vector_23[:, 0] = 0
        vector_23[:, 2] = 0
        res[idx_23] = torch.norm(vector_23, dim=1)

        idx_24 = torch.logical_and(region_xm, torch.logical_and(region_yd, region_zm))
        vector_24 = inputs[idx_24] - torch.FloatTensor((0, -self.size, 0)).cuda()
        vector_24[:, 0] = 0
        vector_24[:, 2] = 0
        res[idx_24] = torch.norm(vector_24, dim=1)

        # not zm
        idx_25 = torch.logical_and(region_xm, torch.logical_and(region_ym, region_zu))
        vector_25 = inputs[idx_25] - torch.FloatTensor((0, 0, self.size)).cuda()
        vector_25[:, 1] = 0
        vector_25[:, 0] = 0
        res[idx_25] = torch.norm(vector_25, dim=1)

        idx_26 = torch.logical_and(region_xm, torch.logical_and(region_ym, region_zd))
        vector_26 = inputs[idx_26] - torch.FloatTensor((0, 0, -self.size)).cuda()
        vector_26[:, 1] = 0
        vector_26[:, 0] = 0
        res[idx_26] = torch.norm(vector_26, dim=1)
        #inside
        idx_27 = torch.logical_and(region_xm, torch.logical_and(region_ym, region_zm))
        vector_27_u = torch.abs(inputs[idx_27] - torch.FloatTensor((self.size, self.size, self.size)).cuda())
        vector_27_d = inputs[idx_27] - torch.FloatTensor((-self.size, -self.size, -self.size)).cuda()
        vector_27 =  torch.cat((vector_27_u,vector_27_d),dim=1)

        res[idx_27] = torch.min(vector_27, dim=1)[0]

        return res

    def udf(self, x):
        return self.forward(x)

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.udf(x)
        # y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

