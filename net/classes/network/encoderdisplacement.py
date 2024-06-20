from typing import Dict, List
from network.network import Network
from network.siren import gradient
import torch
import torch.nn.functional as F

class EncoderDisplacement(Network):

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
        self.features : torch.Tensor = None
        super().__init__(config)

    def _initialize(self):
        self.iteration = 0
        self.epoch = 0
        if self.freeze_base and self.base is not None:
             self.base.requires_grad_(False)

        if(hasattr(self.residual, "base")):
            self.residual.base = self.base

        self.register_buffer('factor', torch.tensor(self.offset_max))


    def eval_base(self,points: torch.Tensor) -> torch.Tensor:
        if(self.base is None):
            return (points.norm(dim=-1) - self.sphere_radius)
        return self.base({"coords":points,"detach":False})["sdf"]

    def eval_grad(self,value,points:torch.Tensor) -> torch.Tensor:
        if(self.base is None):
            return points/(points.norm(dim=-1,keepdim=True))
        grad = gradient(value, points, graph=True)

        return grad
    

    def iniFeatures(self):
        with torch.no_grad():
            temp = self.runner.data.getAllFeatureCoords().transpose(1,2).cuda()
            self.features = torch.empty((temp.shape[0],self.base.featureEncoder.fea_dim)).cuda()
            start = 0
            end = temp.shape[0]
            _batch = 64**2
            while(start < end):
                self.features[start:min(start+_batch-1,end),:], _, _ = self.base.featureEncoder(
                    temp[start:min(start+_batch-1,end),:,:])
                start +=_batch

    def encode(self, args:Dict, _coords=None)->torch.Tensor:
        gridIds = self.runner.data.getGridIds(args.cpu())
        return self.features[gridIds.astype(int)]
        # self.features, _, _ = self.base.featureEncoder(self.runner.data._gridFeaturePoints.transpose(1,2))
        # pass
        

    def evaluate(self, query_coords, fea=None, **args)->Dict[str, torch.Tensor]:
        force_run = args.get("force_run", False)
        is_train = args.get("istrain", self.training)
        self.iteration = args.get('iteration', self.iteration)
        self.epoch = args.get('iteration', self.epoch)
        detach = args.get("detach", True)
        # input_points_t = args.get("coords").reshape(-1, 3)
        featureSize = args.get("featureSize")
        if query_coords is None:
            query_coords = args['coords']
            relative_coords = args.get("relative_coords").reshape(-1, 3)
            query_has_feature = args.get("query_has_feature").reshape(-1, 1)
            query_feature_coords = args.get("featurepoints").reshape(-1, featureSize, 3)

        else:
            query_feature_coords, relative_coords, query_has_feature = self.runner.data.getFeatureCoords(query_coords.cpu())
            query_feature_coords = query_feature_coords.cuda()
            relative_coords = relative_coords.cuda()
            query_has_feature = query_has_feature.cuda()

        input_points_t = query_coords.reshape(-1, 3)
        # if(fea is not None):
        #     input_points_t = torch.cat((input_points_t, fea), 1)
        # device
        outputs = {}

        gt_sdf = None
        if args.get("compute_gt", False):
            # compute groundtruth SDF
            try:
                gt_sdf = self.runner.active_task.data.get_sdf(input_points_t.view(-1, 3)).view(1, -1)
            except Exception:
                gt_sdf = input_points_t.new_ones(input_points_t.shape[:-1]) * -1

            outputs['gt'] = gt_sdf

        if (not detach and not input_points_t.requires_grad):
            detach = True

        with torch.enable_grad():
            base = self.base({"coords": input_points_t, "detach": detach, "fea" : fea,"query_has_feature" : query_has_feature,
                              "query_feature_coords": query_feature_coords, "relative_coords" : relative_coords, "isTrain" : is_train})
            value = base["sdf"]
            relative_coords_t = base["detached"]
            # grad = torch.autograd.grad(value, [input_points_t], grad_outputs=torch.ones_like(value), retain_graph=True,
            #                            create_graph=True)[0]
            grad = gradient(value, relative_coords_t)

        normal = F.normalize(grad, p=2, dim=1)

        # We don't care about area far from the base surface
        if self.close_surface_activation:
            activation = 1.0 / (1 + (value.detach() / self.activation_threshold) ** 4)
        else:
            activation = torch.ones_like(value).detach()

        if not force_run and is_train and not self.residual.requires_grad:
            outputs.update({"sdf": torch.zeros_like(value).detach(), "residual": torch.zeros_like(value).detach(),
                            "detached": relative_coords_t, "activation": activation,
                            "base": value, "base_normal": grad,
                            "prediction": torch.zeros_like(input_points_t)})
            return outputs

        ########## displacement ###########
        if hasattr(self, "use_normal") and self.use_normal:
            if hasattr(self, "scale_base_value") and self.scale_base_value:
                value = torch.tanh(0.8 / self.activation_threshold * value)
            if self.detach_normal:
                residual = \
                self.residual({"coords": torch.cat([normal.detach(), value.reshape(-1, 1)], 1), "detach": False})["sdf"]
            else:
                residual = self.residual({"coords": torch.cat([normal, value.reshape(-1, 1)], 1), "detach": False})[
                    "sdf"]
        else:
            residual = self.residual({"coords": input_points_t, "detach": False})["sdf"]

        # TODO activation based on baseSDF
        if is_train:
            self.factor.fill_(min(self.iteration * self.offset_base, self.offset_max))

        if self.use_tanh:
            residual = torch.tanh(residual)

        residual = self.factor * residual * activation
        prediction = relative_coords_t + residual * normal.detach()
        # query_feature_coords, relative_coords ,query_has_feature= self.runner.data.getFeatureCoords(prediction.cpu())
        # query_feature_coords = query_feature_coords.cuda()
        # relative_coords = relative_coords.cuda()
        # query_has_feature = query_has_feature.cuda()


        result = self.base({"coords": prediction, "detach": False,  "fea" : fea, "query_has_feature" : query_has_feature,
                            "query_feature_coords" : query_feature_coords
                            ,"relative_coords" : prediction, "isTrain" : is_train})["sdf"]

        outputs.update({"sdf": result, "detached": relative_coords_t, "base": value, "residual": residual,
                        "base_normal": grad, "prediction": prediction, "gt": gt_sdf, "activation": activation})
        return outputs
        # kwargs.update({'coords': query_coords})
        # return self.forward(kwargs)

    def forward(self, args):
        # coords = args['coords']
        # self.epoch = args.get("epoch", self.epoch)

        ###### query point feature from the feature net #######
        isEncoding = args.get("isEncoding", False)
        # if (isEncoding and self.base.encoder is not None):
        #     features = self.encode(args)
        # else:
        #     features = None
        outputs = self.evaluate(None, **args)
        return outputs

    def save(self, path):
        torch.save(self, path)
        torch.save(self.base, path[:-4] + "_base" + path[-4:])
