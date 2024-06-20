import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  network.network import Network
from network.sirenudf import MySoftplus



# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# class MySoftplus(nn.Module):
#     def __init__(self,betar:int = 1,betal:int = 1,threshold:int = 1) -> None:
#         super().__init__()
#         self.beta1 = betar
#         self.beta2 = betal
#         self.threshold = threshold

#     def forward(self, input):
#         return torch.where(
#             input>0,torch.nn.functional.softplus(input,beta=self.beta1,threshold=self.threshold),
#             self.beta2/self.beta1*torch.nn.functional.softplus(input,beta=self.beta2)
#         )

class LSUDFModule(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

        # self.act_last = nn.Sigmoid()
        # self.act_last = MySoftplus(betar=1e4,betal=100,threshold=1)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        
        res = x
        res = torch.abs(x)
        # res = self.act_last(x)
        return res / self.scale

    def udf(self, x,detach=True):
        if(detach):
            x=x.clone().detach().requires_grad_(True)
        return self.forward(x),x

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.udf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class LSUDF(Network):
    def __init__(self, config):
        self.d_out = 1
        self.d_in = 3
        self.d_hidden = 256
        self.n_layers = 8
        self.skip_in = [4]
        self.multires = 10
        self.bias = 0.5
        self.scale = 1.0
        self.geometric_init = True
        self.weight_norm = True
        super().__init__(config)

    def _initialize(self):
        self._module = LSUDFModule(d_in=self.d_in,
                 d_out=self.d_out,
                 d_hidden=self.d_hidden,
                 n_layers=self.n_layers,
                 skip_in=self.skip_in,
                 multires=self.multires,
                 bias=self.bias,
                 scale=self.scale,
                 geometric_init=self.geometric_init,
                 weight_norm=self.weight_norm)
        

    def forward(self, args):
        detach = args.get("detach",True)
        input_coords = args["coords"]
        # nearest_coords = args.get("nearest_coords",None)
        c = args.get("x", None)
        if c is not None:
            input_coords = torch.cat([input_coords, c], dim=-1)
        result, detached = self._module.udf(input_coords, detach)


        sigmas = torch.ones_like(result)*0.01
        nearest_sigmas = torch.ones(result.shape[0],10,1).cuda()*0.01
        KNN_sigmas = torch.ones(result.shape[0],10,1).cuda()*0.01
        return {"sdf":result, "detached":detached}


    def evaluate(self, coords, fea=None, **kwargs):
        kwargs.update({'coords': coords, 'x': fea})
        return self.forward(kwargs)

    def save(self, path):
        torch.save(self, path)