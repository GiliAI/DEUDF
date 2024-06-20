from typing import Union, List
from loss.loss import Loss
from loss.siren import gradient
import torch
import torch.nn.functional as F
from network.network import approximate_gradient
import heapq

def Hessian(grad,x):
    hessian = torch.zeros([grad.shape[0],grad.shape[1],grad.shape[1]]).cuda()
    for i in range(grad.shape[1]):
        _grad = grad[:,i:i+1]
        hessian[:,i,:] = torch.autograd.grad(_grad,[x],grad_outputs=torch.ones_like(_grad),
                                             create_graph=True,retain_graph=True)[0]
    return hessian

def MyDet(matrix):
    result1 = matrix[...,0,0]*matrix[...,1,1]*matrix[...,2,2]
    result2 = matrix[...,0,0]*matrix[...,1,2]*matrix[...,2,1]
    result3 = matrix[...,0,1]*matrix[...,1,0]*matrix[...,2,2]
    result4 = matrix[...,0,1]*matrix[...,1,2]*matrix[...,2,0]
    result5 = matrix[...,0,2]*matrix[...,1,0]*matrix[...,2,1]
    result6 = matrix[...,0,2]*matrix[...,1,1]*matrix[...,2,0]
    result = result1 -result2 -result3 +result4+result5 -result6

    return result



class UdfAbs(Loss):
    def __init__(self, config):
        self.sign_loss = False
        self.sdf : Union[List, float] = 3e2
        self.normal : Union[List, float] = 2e2
        self.grad : Union[List, float] = 5e1
        self.inter : Union[List, float] = 1e2
        self.base_sdf : Union[List, float] = 0
        self.base_normal : Union[List, float] = 0
        self.base_inter: Union[List, float] = 0
        self.base_grad: Union[List, float] = 0
        self.offsetR : float = 0.001
        self.hasgtnormal : bool = True
        super().__init__(config)

    def __call__(self, model_output : dict, model_input: dict):
        '''
        x: batch of input coordinates
        y: usually the output of the trial_soln function
        '''
        gt_sdf = model_input["sdf_out"]
        gt_normals = model_input["normal_out"]
        coords = model_output["detached"]


        prediction = model_output["sdf"]
        pred_sdf = prediction.view_as(gt_sdf)


        udf_loss_masks = model_input["masks"].view_as(gt_sdf)

        
        progress = model_input.get('progress', 1.0)

        sdf = self._get_loss_weight('sdf', progress)
        inter =self._get_loss_weight('inter', progress)

        grad = self._get_loss_weight('grad', progress)
        normal = self._get_loss_weight('normal', progress)

        base_sdf = self._get_loss_weight('base_sdf', progress)
        base_inter = self._get_loss_weight('base_inter', progress)
        base_grad = self._get_loss_weight('base_grad', progress)
        base_normal = self._get_loss_weight('base_normal', progress)

        base_hessian = self._get_loss_weight('base_hessian',progress)
        hessian = self._get_loss_weight('hessian',progress)

        zero_base_weight = self._get_loss_weight('zero_base',progress)
        zero_dis_weight = self._get_loss_weight('zero_dis',progress)

        base_grad_lr = self._get_loss_weight('base_grad_lr',progress)
        base_normal_lr = self._get_loss_weight('base_normal_lr',progress)
        grad_lr = self._get_loss_weight('grad_lr',progress)
        normal_lr = self._get_loss_weight('normal_lr',progress)

        offsetRW = self._get_loss_weight('offsetRW',progress)
        offsetR = offsetRW * self.offsetR


        self.runner.logger.log_scalar("loss_weight",
            dict([("weight_sdf", sdf),
            ("weight_inter", inter),
            ("weight_grad", grad),
            ("weight_normal", normal),
            ("weight_base_sdf", base_sdf),
            ("weight_base_inter", base_inter),
            ("weight_base_grad", base_grad),
            ("weight_base_normal", base_normal),
            # ("weight_sdf_moved", sdf_moved),
            # ("weight_reverse",reverse_weight),
            ("weight_zero_base",zero_base_weight),
            ("weight_zero_dis",zero_dis_weight),

            ("weight_base_grad_lr", base_grad_lr),
            ("weight_base_normal_lr",base_normal_lr),
            ("weight_grad_lr", grad_lr),
            ("weight_normal_lr",normal_lr),

            ("offsetRW",offsetRW)
            # ("weight_base_sdf_moved", base_sdf_moved)
            ]))

        approx_grad = getattr(self.runner.network, 'approximate_gradient', False)
        fea = model_output.get('encoded', None)

        value = model_output["base"].view_as(gt_sdf)

        result = {}
        scalar_result = {}

        ############# base #############
        # base sdf
        if 'base' in model_output:
            prediction_base = model_output['base'].view_as(gt_sdf)
            prediction_origin_base = model_output['origin_base'].view_as(gt_sdf)
            if base_sdf > 0:
                base_constraint = torch.where(gt_sdf != -1, torch.abs(prediction_base),
                                            torch.zeros_like(prediction_base)) * udf_loss_masks
                scalar_result['base_sdf_loss'] = base_constraint
                result['base_sdf_loss'] = base_constraint * base_sdf 

            if zero_base_weight >0:
                zero_base_udf = model_output["zero_base_udf"].view_as(gt_sdf)
                zero_base_constraint = torch.where((zero_base_udf<0)&(gt_sdf != -1), torch.abs(zero_base_udf),
                                            torch.zeros_like(zero_base_udf)) 
                scalar_result['zero_base_loss'] = zero_base_constraint
                result['zero_base_loss'] = zero_base_constraint * zero_base_weight 

            # SDF off the surface
            if base_inter > 0:
                inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(prediction_base),
                                                torch.exp(-1e2 * prediction_origin_base))
                scalar_result["base_inter_loss"] = inter_constraint 
                result["base_inter_loss"] = inter_constraint * base_inter

            # base normal direction and eikonal
            if (base_normal > 0 or base_grad >0) and 'base_normal' in model_output:
                gradientV = model_output['base_normal'].view_as(gt_normals)
                base_normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    1 - torch.abs(F.cosine_similarity(gradientV,
                                                            gt_normals, dim=-1)),
                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                
                zero_base_udf = model_output["zero_base_udf"]
                zero_base_coords = model_output["zero_base_coords"]
                zero_base_normal = gradient(zero_base_udf,zero_base_coords).view_as(gt_normals)
                base_zero_normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    1 - torch.abs(F.cosine_similarity(gradientV,
                                                            zero_base_normal, dim=-1)),
                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))

                scalar_result['base_normal_loss'] = base_normal_constraint

                scalar_result['base_normal_zerobase_loss'] = base_zero_normal_constraint

                result['base_normal_loss'] = (base_normal_constraint + base_zero_normal_constraint)* base_normal

                # also apply eikonal
                # onsurface change to grad0
                # base_grad0_constraint = torch.where(~torch.all(gt_normals == -1, dim=-1),
                #                     (gradientV.norm(dim=-1)-1).abs(),torch.zeros(gt_normals.shape[:-1], device=gt_normals.device))
                base_eikonal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device),(gradientV.norm(dim=-1) -1)))
                # scalar_result['base_grad0_loss'] = base_grad0_constraint
                scalar_result['base_grad_loss'] = base_eikonal_constraint
                result['base_grad_loss'] = base_eikonal_constraint * base_grad


                if base_hessian > 0 :
                    base_hessian_matrix = Hessian(model_output['base_normal'],coords)
                    base_hessian_det = MyDet(base_hessian_matrix).view_as(gt_sdf)
                    base_hessian_constraint = torch.where(gt_sdf != -1, torch.abs(base_hessian_det),torch.zeros_like(base_hessian_det))
                    scalar_result['base_grad_loss'] = base_hessian_constraint
                    result['base_hessian_loss'] = base_hessian_constraint * base_hessian

            if (base_normal_lr > 0 or base_grad_lr >0) and 'base_normal' in model_output:
                gradientL = model_output['base_normal_l'].view_as(gt_normals)
                gradientR = model_output['base_normal_r'].view_as(gt_normals)
                if self.hasgtnormal:
                    base_normal_constraint_l = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                        1 + F.cosine_similarity(gradientL,
                                                                gt_normals, dim=-1),
                                        torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                    base_normal_constraint_r = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                        1 - F.cosine_similarity(gradientR,
                                                                gt_normals, dim=-1),
                                        torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                else:
                    base_normal_constraint_l = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                        1 - torch.abs(F.cosine_similarity(gradientL,
                                                                gradientR, dim=-1)),
                                        torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                    base_normal_constraint_r = base_normal_constraint_l
                base_normal_constraint_lr = base_normal_constraint_l/2 + base_normal_constraint_r/2
                scalar_result['base_normal_lr_loss'] =base_normal_constraint_lr
                result['base_normal_lr_loss'] = base_normal_constraint_lr * base_normal_lr


                # also apply eikonal
                base_gradL_constraint = (gradientL.norm(dim=-1) -1).abs()
                base_gradR_constraint = (gradientR.norm(dim=-1) -1).abs()
                _offset_l =  model_output["base_l"].detach().view_as(base_gradL_constraint)
                _offset_l_weight = 1/(1+(offsetR/_offset_l)**4)
                _offset_r =  model_output["base_r"].detach().view_as(base_gradR_constraint)
                _offset_r_weight = 1/(1+(offsetR/_offset_r)**4)
                base_grad_lr_constraint = base_gradL_constraint/2 * _offset_l_weight\
                      + base_gradR_constraint/2 * _offset_r_weight
                scalar_result['base_grad_lr_loss'] = base_grad_lr_constraint
                result['base_grad_lr_loss'] = base_grad_lr_constraint * base_grad_lr

        if sdf > 0:
            sdf_constraint = torch.where(gt_sdf != -1, torch.abs(pred_sdf),
                                         torch.zeros_like(pred_sdf)) * udf_loss_masks

            scalar_result["sdf_loss"] = sdf_constraint
            result["sdf_loss"] = sdf_constraint * sdf
        

        if zero_dis_weight >0:
            zero_dis_udf = model_output["zero_dis_udf"].view_as(gt_sdf)
            zero_dis_constraint = torch.where((zero_dis_udf<0)&(gt_sdf != -1), torch.abs(zero_dis_udf),
                                            torch.zeros_like(zero_dis_udf))
            scalar_result['zero_dis_loss'] = zero_dis_constraint
            result['zero_dis_loss'] = zero_dis_constraint * zero_dis_weight 

        # SDF off the surface
        if inter > 0:
            pred_origin_sdf = model_output['origin_sdf'].view_as(gt_sdf)
            inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_origin_sdf),
                                            torch.exp(-1e2 * pred_origin_sdf)) 
            scalar_result["inter_loss"] = inter_constraint
            result["inter_loss"] = inter_constraint * inter
        

        ############## normal #############
        if grad_lr > 0 or normal_lr > 0:
            coords_l = model_output["detached_l"]
            coords_r = model_output["detached_r"]
            prediction_l = model_output["sdf_l"]
            prediction_r = model_output["sdf_r"]
            gradientL = gradient(prediction_l, coords_l)
            gradientR = gradient(prediction_r, coords_r)

            # eikonal constraint
            if grad_lr > 0:
                gradL_constraint = torch.abs(gradientL.norm(dim=-1) - 1)
                gradR_constraint = torch.abs(gradientR.norm(dim=-1) - 1)
                _offset_l =  model_output["sdf_l"].detach().view_as(gradL_constraint)
                _offset_l_weight = 1/(1+(offsetR/_offset_l)**4)+0.01
                _offset_r =  model_output["sdf_r"].detach().view_as(gradR_constraint)
                _offset_r_weight = 1/(1+(offsetR/_offset_r)**4)+0.01
                grad_constraint = gradL_constraint/2 * _offset_l_weight \
                    + gradR_constraint/2 * _offset_r_weight
                scalar_result['grad_lr_loss'] = grad_constraint
                result['grad_lr_loss'] = grad_constraint * grad_lr


            gradientL = gradientL.view_as(gt_normals)
            gradientR = gradientR.view_as(gt_normals)

            if normal_lr > 0:
                if self.hasgtnormal :
                    normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                                    1 + F.cosine_similarity(gradientL,gt_normals, dim=-1),
                                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))/2+\
                                        torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                                    1 - F.cosine_similarity(gradientR,gt_normals, dim=-1),
                                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))/2
                else : 
                    normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                                    1 - torch.abs(F.cosine_similarity(gradientL,gradientR, dim=-1)),
                                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                scalar_result["normal_lr_loss"] = normal_constraint
                result["normal_lr_loss"] = normal_constraint * normal_lr
        
        if grad > 0 or normal > 0:
            if approx_grad:
                gradientV = approximate_gradient(self.runner.network, coords.detach(), fea=fea)
            else:
                gradientV = gradient(prediction, coords)
            gradientV = gradientV.view_as(gt_normals)

            # eikonal constraint
            # change to grad0 constraint
            if grad > 0:
                grad0_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    (gradientV.norm(dim=-1)-1),torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                grad_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device),(gradientV.norm(dim=-1) -1)))
                _offset =  pred_sdf.detach().view_as(grad_constraint)
                _offset_weight = 1/(1+(offsetR/_offset)**4)+0.01
                scalar_result['grad0_loss'] = grad0_constraint
                scalar_result['grad_loss'] = grad_constraint
                result['grad_loss'] = (grad_constraint+grad0_constraint*_offset_weight) * grad


                if hessian > 0 :
                    hessian_matrix = Hessian(model_output['base_normal'],coords)
                    hessian_det = MyDet(hessian_matrix).view_as(gt_sdf)
                    hessian_constraint = torch.where(gt_sdf != -1, torch.abs(hessian_det),torch.zeros_like(hessian_det))
                    scalar_result['hessian_loss'] = hessian_constraint
                    result['hessian_loss'] = hessian_constraint * hessian

            # normal direction for the on-surface points

            if normal > 0:
                normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                                1 - torch.abs(F.cosine_similarity(gradientV,gt_normals, dim=-1)),
                                                torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                
                zero_dis_udf = model_output["zero_dis_udf"]
                zero_dis_coords = model_output["zero_dis_coords"]
                zero_dis_normal = gradient(zero_dis_udf,zero_dis_coords).view_as(gt_normals)
                zero_normal_constraint = torch.abs(torch.where(~torch.all(gt_normals == -1, dim=-1),
                                    1 - torch.abs(F.cosine_similarity(gradientV,
                                                            zero_dis_normal, dim=-1)),
                                    torch.zeros(gt_normals.shape[:-1], device=gt_normals.device)))
                scalar_result["normal_loss"] = normal_constraint
                scalar_result["zero_normal_loss"] = zero_normal_constraint

                result["normal_loss"] = (normal_constraint + zero_normal_constraint) * normal

        


        if(self.sign_loss):
            sign = model_output["normal"].reshape([-1,3])
            sign_constraint =  1 - F.cosine_similarity(sign,
                                                       gt_normals, dim=-1)
            scalar_result['sign_loss'] = sign_constraint
            result["sign_loss"] = sign_constraint

        # return result
        return result, scalar_result