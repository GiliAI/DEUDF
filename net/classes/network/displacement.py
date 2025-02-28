from network.network import Network
from network.siren import gradient
import torch
import torch.nn.functional as F

class Displacement(Network):

    def __init__(self, config):
        self.base : Network = None
        self.residual : Network = None
        self.freeze_base : bool = False
        self.offset_base : float = 0.02
        self.offset_max : float = 0.1
        self.use_normal : bool = False
        self.detach_normal: bool = False
        self.close_surface_activation: bool = True
        self.activation_threshold : float = 0.05
        self.use_tanh :bool = True
        super().__init__(config)

    def _initialize(self):
        self.iteration = 0
        self.epoch = 0
        if self.freeze_base and self.base is not None:
             self.base.requires_grad_(False)

        if(hasattr(self.residual, "base")):
            self.residual.base = self.base

        self.register_buffer('factor', torch.tensor(self.offset_max))
        # for i in range(self.base.hidden_layers+2):
        #     setattr(self,"base"+str(i),self.base._module.net[i])

    def eval_base(self,points: torch.Tensor) -> torch.Tensor:
        if(self.base is None):
            return (points.norm(dim=-1) - self.sphere_radius)
        return self.base({"coords":points,"detach":False})["sdf"]

    def eval_grad(self,value,points:torch.Tensor) -> torch.Tensor:
        if(self.base is None):
            return points/(points.norm(dim=-1,keepdim=True))
        grad = gradient(value,points, graph=True)

        return grad

    def encode(self, *args, **kwargs):
        pass

    def evaluate(self, query_coords, fea=None, **kwargs):
        # kwargs.update({'coords': query_coords,'is_eval':True})
        # KNN_coords, Distance = self.runner.data.getKNNcoords(query_coords)
        query_coords.requires_grad_(True)
        kwargs.update({'coords': query_coords,
                        # 'coords_lr': torch.zeros_like(query_coords),
                       'coords_l': torch.zeros_like(query_coords),
                        'coords_r': torch.zeros_like(query_coords),
                       'is_eval':True,'istrain':False})
        # kwargs.update({'coords': query_coords,'is_eval':True,"KNN_coords":KNN_coords.cuda(),"Distance":Distance.cuda()})
        # coords = query_coords.clone().detach().requires_grad_(True)
        return self.forward(kwargs)

    def forward(self, args):
        force_run = args.get("force_run", False)
        is_train = args.get("istrain", self.training)
        self.iteration = args.get('iteration', self.iteration)
        self.epoch = args.get('iteration', self.epoch)
        detach = args.get("detach", True)
        is_eval = args.get("is_eval",False)
        input_points_t = args.get("coords").reshape(-1,3)
        # input_points_lr = args.get("coords_lr",torch.zeros_like(input_points_t)).reshape(-1,3)
        input_points_l = args.get("coords_l",torch.zeros_like(input_points_t)).reshape(-1,3)
        input_points_r = args.get("coords_r",torch.zeros_like(input_points_t)).reshape(-1,3)
        zero_base_coords = args.get("zero_base_coords",torch.zeros_like(input_points_t)).reshape(-1,3)
        # KNN_coords = args.get("KNN_coords").reshape(-1,3)
        # Distance = args.get("Distance").reshape(-1,1)

        # if is_eval:
        #     input_nearest_points_t = None
        # else:
        #     input_nearest_points_t = args.get("nearest_coords").reshape(-1,3)

        
        # if is_eval:
        #     input_points_t = args.get("coords")
        # else:
        #     input_points_t = args.get("coords").reshape(-1,3)

        outputs = {}

        gt_sdf = None
        if args.get("compute_gt", False):
            # compute groundtruth SDF
            try:
                gt_sdf = self.runner.active_task.data.get_sdf(input_points_t.view(-1, 3)).view(1,-1)
            except Exception:
                gt_sdf = input_points_t.new_ones(input_points_t.shape[:-1])*-1

            outputs['gt'] = gt_sdf


        if (not detach and not input_points_t.requires_grad):
            detach = True

        with torch.enable_grad():
            # hessian_metrix = torch.autograd.functional.hessian(lambda x: self.base({"coords":x,"nearest_coords":input_nearest_points_t,"detach":detach})["sdf"],input_points_t)
            base = self.base({"coords":input_points_t,
                            #   "nearest_coords":input_nearest_points_t,
                              # "KNN_coords":KNN_coords,
                              "detach":detach})
            value = base["sdf"]
            input_points_t = base["detached"]
            # sdf_moved = base["sdf_moved"]
            # sigmas = base["sigmas"]
            # nearest_sigmas = base["nearest_sigmas"]
            # coords_moved = base["coords_moved"]
            # KNN_sigmas = base["KNN_sigmas"]
            # nearest_input_points = self.runner.data.getNearestPoint(input_points_t).reshape(-1,1)
            # nearest_sigmas = self.base({"coords":input_nearest_points_t, "detach":detach})["sigmas"].\
            #     reshape(input_points_t.shape[0],-1,1)
            # grad = torch.autograd.grad(torch.abs(value), [input_points_t], grad_outputs=torch.ones_like(value),retain_graph=True,
            #                         create_graph=True)[0]
            grad = torch.autograd.grad(value, [input_points_t], grad_outputs=torch.ones_like(value),retain_graph=True,
                                    create_graph=True)[0]
            if is_train:
                # base_lr = self.base({"coords":input_points_lr,"detach":detach})
                # input_points_lr = base_lr["detached"]
                # grad_lr = torch.autograd.grad(base_lr["sdf"], [input_points_lr], grad_outputs=torch.ones_like(value),retain_graph=True,
                #                         create_graph=True)[0]
                base_l = self.base({"coords":input_points_l,"detach":detach})
                input_points_l = base_l["detached"]
                value_l = base_l["sdf"]
                grad_l = torch.autograd.grad(value_l, [input_points_l], grad_outputs=torch.ones_like(value),retain_graph=True,
                                        create_graph=True)[0]
                base_r = self.base({"coords":input_points_r,"detach":detach})
                value_r = base_r["sdf"]
                input_points_r = base_r["detached"]
                grad_r = torch.autograd.grad(value_r, [input_points_r], grad_outputs=torch.ones_like(value),retain_graph=True,
                                        create_graph=True)[0]
                zero_base = self.base({"coords":zero_base_coords,
                                #   "nearest_coords":input_nearest_points_t,
                                # "KNN_coords":KNN_coords,
                                "detach":detach})
                zero_base_udf = zero_base["sdf"]
                zero_base_coords = zero_base["detached"]
            else :
                # base_lr = torch.zeros_like(value)
                # grad_lr = torch.zeros_like(grad)
                value_l = torch.zeros_like(value)
                grad_l = torch.zeros_like(grad)
                value_r = torch.zeros_like(value)
                grad_r = torch.zeros_like(grad)
                zero_base_udf = torch.zeros_like(value)
                zero_base_coords = torch.zeros_like(input_points_t)


        normal = F.normalize(grad, p=2, dim=1)
        # normal_lr = F.normalize(grad_lr, p=2, dim=1)
        normal_l = F.normalize(grad_l, p=2, dim=1)
        normal_r = F.normalize(grad_r, p=2, dim=1)


        # We don't care about area far from the base surface
        if self.close_surface_activation:
            activation = 1.0 / (1 + (value.detach()/self.activation_threshold)**4)
        else:
            activation = torch.ones_like(value).detach()

        # sdf_moved = sdf_moved * activation

        if not force_run and is_train and not self.residual.requires_grad:
            outputs.update({"sdf":torch.zeros_like(value).detach(), "residual":torch.zeros_like(value).detach(),
                    "detached":input_points_t, "activation": activation,
                    # "origin_base":value,"base":torch.abs(value), 
                    "origin_base":value,"base":value, 
                    "base_normal": grad,
                    "base_l" : value_l,
                    "base_r" : value_r,
                    "base_normal_l":grad_l,
                    "base_normal_r":grad_r,
                    # "base_normal_lr":grad_lr,
                    # "base_coords_moved":coords_moved,
                    "prediction":torch.zeros_like(input_points_t),
                    "zero_base_udf":zero_base_udf,
                    "zero_base_coords" : zero_base_coords
                    # ,"sdf_moved":sdf_moved,"sigmas":sigmas,"nearest_sigmas":nearest_sigmas
                    })
            # outputs.update({"sdf":torch.zeros_like(value).detach(), "residual":torch.zeros_like(value).detach(),
            #         "detached":input_points_t, "activation": activation,
            #         "base":value, "base_normal": grad,
            #         "prediction":torch.zeros_like(input_points_t)})
            return outputs

        ########## displacement ###########
        if hasattr(self, "use_normal") and self.use_normal:
            if hasattr(self, "scale_base_value") and self.scale_base_value:
                value = torch.tanh(0.8/self.activation_threshold*value)
            if self.detach_normal:
                residual = self.residual({"coords":torch.cat([normal.detach(), value.reshape(-1,1)], 1), "detach":False})["sdf"]
            else:
                residual = self.residual({"coords":torch.cat([normal, value.reshape(-1,1)], 1), "detach":False})["sdf"]
        else:
            residual = self.residual({"coords":input_points_t, "detach":False})["sdf"]
            if is_train:
                # residual_lr = self.residual({"coords":input_points_lr, "detach":False})["sdf"]
                residual_l = self.residual({"coords":input_points_l, "detach":False})["sdf"]
                residual_r = self.residual({"coords":input_points_r, "detach":False})["sdf"]
            else :
                # residual_lr = torch.zeros_like(residual)
                residual_l = torch.zeros_like(residual)
                residual_r = torch.zeros_like(residual)


        # TODO activation based on baseSDF
        if is_train:
            self.factor.fill_(min(self.iteration*self.offset_base, self.offset_max))
        # project residual to (-1,1)
        if self.use_tanh:
            residual = torch.tanh(residual)
            # residual_lr = torch.tanh(residual_lr)
            residual_l = torch.tanh(residual_l)
            residual_r = torch.tanh(residual_r)



        # residual = self.factor*residual*activation
        residual = self.offset_max*residual*activation
        # residual_lr = self.offset_max*residual_lr*activation
        activation_l = 1.0 / (1 + (value_l.detach()/self.activation_threshold)**4)
        activation_r = 1.0 / (1 + (value_r.detach()/self.activation_threshold)**4)
        residual_l = self.offset_max*residual_l*activation_l
        residual_r = self.offset_max*residual_r*activation_r

        # residual = self.factor*residual*activation
        # test_KNN, test_Distance = self.runner.data.getKNNcoords(torch.tensor([[-0.271961,-0.114314,0.146265],
        #                                                                       [-0.271961,-0.122196,0.151115]]))
        # test_KNNsigmas = self.base({"coords":torch.zeros_like(test_KNN).cuda(),
        #                       "KNN_coords":test_KNN.cuda(),
        #                       "detach":detach})["KNN_sigmas"]
        # test_act = torch.exp(torch.mean((1-1e2*test_KNNsigmas),1)*torch.exp(-1e2*test_Distance.cuda()))

        # new_activation = torch.exp(torch.mean((1-1e2*KNN_sigmas),1)*torch.exp(-1e2*Distance))
        # moved_inputs_points = input_points_t - value*normal 
        # new_activation = torch.exp((1-1e2*sigmas))
        # input_points_t = input_points_t*new_activation + moved_inputs_points*(1-new_activation)

        # base = self.base({"coords":input_points_t,"nearest_coords":input_nearest_points_t,
        #                       # "KNN_coords":KNN_coords,
        #                       "detach":detach})
        # value = base["sdf"]
        # input_points_t = base["detached"]
        # sdf_moved = base["sdf_moved"]
        # sigmas = base["sigmas"]
        # nearest_sigmas = base["nearest_sigmas"]
        # grad = torch.autograd.grad(value, [input_points_t], grad_outputs=torch.ones_like(value),retain_graph=True,
        #                             create_graph=True)[0]
        # new_activation = torch.exp((1-1e2*sigmas))
        # new_activation = Distance * torch.exp(torch.mean((1-1e2*KNN_sigmas),1))
        # new_activation = Distance
        # residual = self.factor*residual*new_activation
        # residual = self.residual({"coords":input_points_t, "detach":False})["sdf"]
        # residual = self.factor*residual
        # residual = residual*new_activation

        # _normal = torch.where(normal[...,1:2]>0,normal,-1*normal)
        # prediction = input_points_t + residual*_normal.detach()

        prediction = input_points_t + residual*normal.detach()

        # prediction_lr = input_points_lr + residual_lr*normal_lr.detach()

        # prediction_l = input_points_l + residual_l*normal_l.detach()
        # prediction_r = input_points_r + residual_r*normal_r.detach()
        # allResult = self.base({"coords":prediction, "detach":False})
        #                     #    ,"nearest_coords":input_nearest_points_t})
        # result = allResult["sdf"]
        # if is_train:
        #     result_l = self.base({"coords":prediction_l, "detach":False})["sdf"]
        #     result_r = self.base({"coords":prediction_r, "detach":False})["sdf"]
        #     # result_lr = self.base({"coords":prediction_lr, "detach":False})["sdf"]
        # else :
        #     result_l = torch.zeros_like(result)
        #     result_r = torch.zeros_like(result)
        #     # result_lr = torch.zeros_like(result)

        result = value + residual
        if is_train:
            result_l = value_l + residual_l
            result_r = value_r + residual_r
        else :
            result_l = torch.zeros_like(result)
            result_r = torch.zeros_like(result)

        residual_zerobase = self.residual({"coords":zero_base_coords, "detach":False})["sdf"]
        _activation = 1.0 / (1 + (zero_base_udf.detach()/self.activation_threshold)**4)
        _activation = _activation.to(residual_zerobase.device)
        residual_zerobase = self.offset_max*residual_zerobase*_activation
        

        zero_dis_coords = args.get("zero_dis_coords",torch.zeros_like(input_points_t)).reshape(-1,3)
        if is_train:
            zero_dis = self.base({"coords":zero_dis_coords, "detach":detach})
            _zero_dis_udf = zero_dis["sdf"]
            zero_dis_coords = zero_dis["detached"]
            residual_zerodis = self.residual({"coords":zero_dis_coords, "detach":False})["sdf"]
            __activation = 1.0 / (1 + (_zero_dis_udf.detach()/self.activation_threshold)**4)
            __activation = __activation.to(residual_zerodis.device)
            residual_zerodis = self.offset_max*residual_zerodis*__activation
            zero_dis_udf = _zero_dis_udf + residual_zerodis
        else:
            zero_dis_udf = torch.zeros_like(result)

        # grad_result = torch.autograd.grad(result, [input_points_t], grad_outputs=torch.ones_like(value),retain_graph=True,
        #                             create_graph=True)[0]
        # grad_dir = torch.bmm(grad_result,grad)
        # activation2 = 
            
                            #    ,"nearest_coords":input_nearest_points_t})
        # _result = value + residual
        # result = self.lastActive(_result)

        # result = torch.nn.functional.relu(_result)
        # result = torch.abs(_result)

        # sdf_moved = allResult["sdf_moved"]
        # sigmas = allResult["sigmas"]
        # sdf_moved = sdf_moved * activation

        outputs.update({"sdf":result, "origin_sdf":result,
                        # "sdf":torch.abs(result), "origin_sdf":result,
                        "detached":input_points_t,

                        # "sdf_lr":result_lr,
                        # "detached_lr":input_points_lr,
                        # "base_normal_lr":grad_lr,
                        "base_l" : value_l,"base_r" : value_r,
                        "sdf_l":result_l,"sdf_r":result_r,
                        "detached_l":input_points_l,
                        "detached_r":input_points_r,
                        "base_normal_r":grad_r,
                        "base_normal_l":grad_l,
                        "residual_zerobase":residual_zerobase,

                        # "origin_base":value,"base":torch.abs(value), 
                        "origin_base":value,"base":value, 
                        "residual":residual,
                        "base_normal":grad, "prediction":prediction, "gt":gt_sdf, "activation": activation,
                        "zero_base_udf":zero_base_udf,
                        "zero_base_coords" : zero_base_coords,
                        "zero_dis_udf":zero_dis_udf,
                        "zero_dis_coords" : zero_dis_coords

                # "sdf_moved":sdf_moved,
                # "base_coords_moved":coords_moved,
                # "sigmas":sigmas,"nearest_sigmas":nearest_sigmas
                })
        # outputs.update({"sdf":result, "detached":input_points_t, "base":value, "residual":residual,
        #         "base_normal":grad, "prediction":prediction, "gt":gt_sdf, "activation": activation})
        return outputs

    def save(self, path):
        torch.save(self, path)
        torch.save(self.base, path[:-4] + "_base" + path[-4:])
