import torch
from  network.network import Network
from torch import nn
import numpy as np
from torch import Tensor


def gradient(y, x, grad_outputs=None, graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,retain_graph=True,
                               create_graph=graph)[0]
    return grad


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussi
    # on of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies t
    # he activations before the
    # nonlinearity. Different signals may require different omega_0 in the first
    #  layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to
    # keep the magnitude of
    # activations constant, but boost gradients to the weight matrix
    # (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.dim = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.dim,
                                            1 / self.dim)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.dim) / self.omega_0,
                                            np.sqrt(6 / self.dim) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class MySoftplus(nn.Module):
    def __init__(self,betar:int = 1,betal:int = 1,threshold:int = 1) -> None:
        super().__init__()
        self.beta1 = betar
        self.beta2 = betal
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return torch.where(
            input>0,torch.nn.functional.softplus(input,beta=self.beta1,threshold=self.threshold),
            self.beta2/self.beta1*torch.nn.functional.softplus(input,beta=self.beta2)
        )

class SirenModule(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, lastActive, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
            
        self.lastActive = lastActive
        # addition
        # self.net.append(nn.ReLU())
        # self.net.append(nn.Softplus(beta=130,threshold=100))
        # self.net.append(MySoftplus(beta=100,threshold=1))
        
        # self.lastActive = nn.Softplus(beta=1000,threshold=1)
        # self.lastActive = nn.functional.softplus(beta=1000,threshold=1)
        # self.lastActive = MySoftplus(betar=1e4,betal=100,threshold=1)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, detach=True):
        if(detach):
            # allows to take derivative w.r.t. input
            coords = coords.clone().detach().requires_grad_(True)
        _output = self.net(coords)
        # output = self.lastActive(_output)
        # output = _output

        # output = torch.abs(_output)
        # output = output * 2
        output = nn.functional.softplus(_output,beta=100)

        # output = self.lastActive(_output[..., :1])
        # sigmas = torch.nn.functional.softplus(_output[..., 1:], beta=100.0)+1e-2
        # output = torch.cat([output, sigmas], dim=-1)

        # output = output / 2

        # addition
        # output = torch.abs(output)
        return output, coords



class SirenUdf(Network):

    def __init__(self, config):
        self.omega :  float = 15.0
        self.omega2 : float = 15.0
        self.hidden_size : int = 256
        self.hidden_layers : int = 3
        self.dim : int = 3
        self.out_dim : int = 1
        self.outermost_linear : bool = True
        self._module : nn.Module = None
        self._sigmasModule : nn.Module = None
        self.offset : float = 0.005
        super().__init__(config)

    def _initialize(self):
        self._module = SirenModule(in_features=self.dim, out_features=self.out_dim,
                        hidden_features=self.hidden_size, hidden_layers=self.hidden_layers,
                        lastActive=MySoftplus(betar=1e4,betal=100,threshold=1),
                        outermost_linear=self.outermost_linear,
                        first_omega_0=self.omega,
                        hidden_omega_0=self.omega)
        # self._sigmasModule =  SirenModule(in_features=self.dim, out_features=self.out_dim,
        #                 hidden_features=self.hidden_size, hidden_layers=self.hidden_layers,
        #                 lastActive=lambda x :torch.nn.functional.softplus(x, beta=100.0)+1e-2,
        #                 outermost_linear=self.outermost_linear,
        #                 first_omega_0=self.omega,
        #                 hidden_omega_0=self.omega)

    def forward(self, args):
        detach = args.get("detach",True)
        input_coords = args["coords"]
        # nearest_coords = args.get("nearest_coords",None)
        c = args.get("x", None)
        if c is not None:
            input_coords = torch.cat([input_coords, c], dim=-1)
        result, detached = self._module(input_coords, detach)

        # sigmas, _ = self._sigmasModule(input_coords, detach)
        # udf_sigma, detached = self._module(input_coords, detach)
        # result = udf_sigma[:,:1]

        # input_coords_move = input_coords - result * gradient(result, detached)
        # sdf_moved, _ = self._module(input_coords_move, detach)

        # sdf_moved = sdf_moved[:,:1]

        # nearest_sigmas = torch.zeros([result.shape[0],10,1]).cuda()
        # if(nearest_coords!=None):
        #     _nearest_sigmas ,_ = self._sigmasModule(nearest_coords,False)
        #     # nearest_udf_sigmas ,_ = self._sigmasModule(nearest_coords,False)
        #     # nearest_sigmas = nearest_udf_sigmas[:,1:]
        #     nearest_sigmas = _nearest_sigmas.reshape(result.shape[0],-1,1)
        # else:
        #     nearest_sigmas = None
        # sigmas = torch.ones_like(result)*0.01
        # nearest_sigmas = torch.ones(result.shape[0],10,1).cuda()*0.01
        # KNN_sigmas = torch.ones(result.shape[0],10,1).cuda()*0.01
        return {"sdf":result, "detached":detached
                # , "sdf_moved":sdf_moved,
                #                 "sigmas":sigmas,"nearest_sigmas":nearest_sigmas,"KNN_sigmas":KNN_sigmas
                                }
        # return {"sdf":result, "detached":detached, "sdf_moved":sdf_moved,
        #         "sigmas":udf_sigma[:,1:],"nearest_sigmas":nearest_sigmas}
        # return {"sdf":result, "detached":detached, "sdf_moved":sdf_moved, "sigmas":torch.ones_like(result)}
        # return {"sdf":result, "detached":detached}


    def evaluate(self, coords, fea=None, **kwargs):
        kwargs.update({'coords': coords, 'x': fea})
        return self.forward(kwargs)

    def save(self, path):
        torch.save(self, path)

    def frozeSigma(self):
        self._sigmasModule.requires_grad_(False)
        self._sigmasModule.train(False)
        self.omega = self.omega2
