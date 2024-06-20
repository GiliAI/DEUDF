import torch
from  network.network import Network
from torch import nn
import torch.nn.functional as F
import numpy as np


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




class PosEncoder(nn.Module):
    def __init__(self, inputdim, multries, log_sampling=True):
        # encode para
        super().__init__()

        self.inputdim = inputdim
        self.middim: int = 0
        self.include_input : bool =True
        self.multries : int = multries
        self.log_sampling : bool = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]
        self.embed_fns = []

        self.initEncode()

    def initEncode(self):
        embed_fns = []
        d = self.inputdim
        middim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            middim += d

        max_freq = self.multries

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq-1, steps=max_freq)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (max_freq-1), steps=max_freq)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                middim += d

        self.embed_fns = embed_fns
        self.middim = middim
    def forward(self, inputs):

        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




class STN3d(nn.Module):
    def __init__(self,feadim,isTest):
        super(STN3d, self).__init__()
        self.isTest = isTest
        self.feadim = feadim
        if(self.isTest):
            self.conv1 = torch.nn.Conv1d(3, 16, 1)
            self.conv2 = torch.nn.Conv1d(16, 16, 1)
            self.conv3 = torch.nn.Conv1d(16, 32, 1)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, 9)
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(16)
            self.bn3 = nn.BatchNorm1d(32)
            self.bn4 = nn.BatchNorm1d(16)
            self.bn5 = nn.BatchNorm1d(16)

        else:
            # self.conv1 = torch.nn.Conv1d(3, 64, 1)
            # self.conv2 = torch.nn.Conv1d(64, 128, 1)
            # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            # self.fc1 = nn.Linear(1024, 512)
            # self.fc2 = nn.Linear(512, 256)
            # self.fc3 = nn.Linear(256, 9)
            # self.bn1 = nn.BatchNorm1d(64)
            # self.bn2 = nn.BatchNorm1d(128)
            # self.bn3 = nn.BatchNorm1d(1024)
            # self.bn4 = nn.BatchNorm1d(512)
            # self.bn5 = nn.BatchNorm1d(256)
            self.conv1 = torch.nn.Conv1d(3, 32, 1)
            self.conv2 = torch.nn.Conv1d(32, 64, 1)
            self.conv3 = torch.nn.Conv1d(64, 512, 1)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 9)
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(512)
            self.bn4 = nn.BatchNorm1d(256)
            self.bn5 = nn.BatchNorm1d(128)


        self.activation = nn.Softplus(100)




    def forward(self, x):
        batchsize = x.size()[0]
        x = x.float()
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feadim)


        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        # self.conv1 = torch.nn.Conv1d(k, 64, 1)
        # self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k*k)
        self.conv1 = torch.nn.Conv1d(k, 8, 1)
        self.conv2 = torch.nn.Conv1d(8, 16, 1)
        self.conv3 = torch.nn.Conv1d(16, 32, 1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, k*k)
        self.relu = nn.ReLU()

        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(8)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        x = x.view(-1, 32)


        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, fea_dim, isTest, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.fea_dim = fea_dim
        self.isTest = isTest
        self.stn = STN3d(self.fea_dim, self.isTest)
        if(self.isTest):
            self.conv1 = torch.nn.Conv1d(3, 8, 1)
            self.conv2 = torch.nn.Conv1d(8, 16, 1)
            self.conv3 = torch.nn.Conv1d(16, self.fea_dim, 1)
            self.bn1 = nn.BatchNorm1d(8)
            self.bn2 = nn.BatchNorm1d(16)
            self.bn3 = nn.BatchNorm1d(self.fea_dim)
        else:
            # self.conv1 = torch.nn.Conv1d(3, 64, 1)
            # self.conv2 = torch.nn.Conv1d(64, 128, 1)
            # self.conv3 = torch.nn.Conv1d(128, self.fea_dim, 1)
            # self.bn1 = nn.BatchNorm1d(64)
            # self.bn2 = nn.BatchNorm1d(128)
            # self.bn3 = nn.BatchNorm1d(self.fea_dim)
            self.conv1 = torch.nn.Conv1d(3, 32, 1)
            self.conv2 = torch.nn.Conv1d(32, 64, 1)
            self.conv3 = torch.nn.Conv1d(64, self.fea_dim, 1)
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(self.fea_dim)

        self.activation = nn.Softplus(100)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # zero_Ids = torch.where(x.mean(dim=2).mean(dim=1)==0)[0]

        n_pts = x.size()[2]
        x = x.float()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.activation(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]

        # x = x.view(-1, 1024)
        x = x.view(-1, self.fea_dim)

        # x[zero_Ids] = 0
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.fea_dim, 1).repeat(1, 1, n_pts)
            # x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat





class SirenModule(nn.Module):
    def __init__(self, in_features,fea_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=False,isEncoding=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        #PosEncoder
        self.multries : int = 10
        self._encoder = PosEncoder(inputdim=in_features,multries= self.multries)
        self.middim = self._encoder.middim
        if isEncoding:
            pass
            #self.middim *= 2

        self.net = []
        self.net.append(SineLayer(fea_features+self.middim, hidden_features,
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

        # activation
        self.net.append(nn.Softplus(100))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, relative_coords=None, features=None, detach=True):
        if(features == None):
            if(detach):
                # allows to take derivative w.r.t. input
                coords = coords.clone().detach().requires_grad_(True)
            output = self._encoder(coords)
            output = self.net(output)
            return output, coords
        else:
            if(detach):
                # allows to take derivative w.r.t. input
                relative_coords = relative_coords.clone().detach().requires_grad_(True)
            # rel_coords_fea = self._encoder(relative_coords)
            # output = torch.cat([output,rel_coords_fea], dim=1)
            output = self._encoder(relative_coords)
            output = torch.cat([output,features], dim=1)
            output = self.net(output)
            return output, relative_coords

        # addition
        # output=abs(output)




class SirenFeatureUdf(Network):

    def __init__(self, config):
        self.omega :  float = 15.0
        self.hidden_size : int = 256
        self.hidden_layers : int = 3
        self.inputdim : int = 3
        self.feadim : int = 512
        self.out_dim : int = 1
        self.outermost_linear : bool = True
        self._module : nn.Module = None

        self.featureEncoder : nn.Module = None
        self.featureCoordsNum :int = 512
        self.isEncoding : bool = False
        # self.voxelNum : int = 1

        self.isTest : bool = True




        super().__init__(config)


    def _initialize(self):
        self._module = SirenModule(in_features=self.inputdim,fea_features=self.feadim, out_features=self.out_dim,
                        hidden_features=self.hidden_size, hidden_layers=self.hidden_layers,
                        outermost_linear=self.outermost_linear,isEncoding=self.isEncoding, 
                        first_omega_0=self.omega,
                        hidden_omega_0=self.omega)
        # self.featureCoordsNum = self.runner.data.featureSize * self.runner.data.voxelNum

        self.featureEncoder = PointNetfeat(self.feadim,self.isTest)
        # self.featureEncoder.requires_grad_(False)

    def forward(self, args):
        detach = args.get("detach",True)
        input_coords = args["coords"]
        # input_coords = self._encoder(input_coords)
        # c = args.get("x", None)
        # if c is not None:
        #     input_coords = torch.cat([input_coords, c], dim=-1)
        # fea = args.get("feature", None)
        # if fea is not None:
        #     input_coords = torch.cat([input_coords, fea], dim=-1)
        relative_coords = args.get("relative_coords",None)
        input_fea_coords = args.get("query_feature_coords", None)
        query_has_feature = args.get("query_has_feature", None)
        isTrain = args.get("isTrain", True)
        fea = args.get("fea", None)

        if self.isEncoding and (input_fea_coords is not None):
            input_fea_coords = input_fea_coords.transpose(2,1)
            query_nofea_Ids = torch.where(query_has_feature==0)

            if fea is None:
                self.featureEncoder.requires_grad_(False)
                with torch.no_grad():
                    fea, _, _ = self.featureEncoder(input_fea_coords)
            
            result, detached = self._module(input_coords, relative_coords, fea ,detach=detach)
            result[query_nofea_Ids] = 1
        else:
            result, detached = self._module(input_coords, detach=detach)


        return {"sdf":result, "detached":detached}



    def evaluate(self, coords, fea=None, **kwargs):
        kwargs.update({'coords': coords, 'x': fea})
        return self.forward(kwargs)

    def save(self, path):
        torch.save(self, path)
