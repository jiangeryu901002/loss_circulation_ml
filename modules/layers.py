import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
from utils import projection_norm_inf, projection_norm_inf_and_1, SparseDropout, DropPath, Identity
from IGNNs.functions import ImplicitFunction
from functools import partial


class ImplicitGraph(Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 #+ support_2
        return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)

class GNNLayer(Module):
    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(GNNLayer, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct
        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        X_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        X_1 = torch.spmm(torch.transpose(A, 0, 1), X_1.T).T
        X_2 = phi(X_1)
        X_2 = torch.spmm(torch.transpose(X_2, 0, 1), self.W.T).T
        X_2 = torch.spmm(A, X_2.T).T
        X = X_1 + X_2
        return X

class HyperGNNLayer(Module):
    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(HyperGNNLayer, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

    def forward(self, X_0, A, U, phi, W, Omega_1, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        X_1 = torch.spmm(torch.transpose(U, 0, 1), Omega_1.T).T
        X_1 = torch.spmm(torch.transpose(A, 0, 1), X_1.T).T
        X_2 = phi(X_1)
        X_2 = torch.spmm(torch.transpose(X_2, 0, 1), W.T).T
        X_2 = torch.spmm(A, X_2.T).T
        X = X_1 + X_2
        return X


class HyperImplicitGraph(Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(HyperImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

    def forward(self, X_0, A, U, phi, W, Omega_1, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            W = projection_norm_inf(W, kappa=self.k/A_rho)
        support_1 = torch.spmm(torch.transpose(U, 0, 1), Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        B = support_1 #+ support_2

        A = A if A_orig is None else A_orig
        # def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fw_mitr, compute_dphi=False)
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X
        # return ImplicitFunction.apply(W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)

class Mlp(Module):
    """ MLP Layer
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Conv(Module):
    """ MLP Layer
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1,1))
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1,1))
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MixerBlock(Module):
    """ Residual Block w/ token mixing and channel MLPs
    """
    def __init__(
        self,
        dim,
        seq_len,
        device,
        mlp_ratio=(0.5, 4.0),
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        tokens_dim = int(mlp_ratio[0] * seq_len)
        channels_dim = int(mlp_ratio[1] * dim)
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len,
                                    out_features=tokens_dim,
                                    act_layer=act_layer,
                                    drop=drop)
        self.drop_path = DropPath(device=device, drop_prob=drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim,
                                      out_features=channels_dim,
                                      act_layer=act_layer,
                                      drop=drop)

    def forward(self, x):
        x = self.drop_path(
            self.mlp_tokens(self.norm1(x).permute(0, 2, 1)).permute(0, 2, 1))
        x = self.drop_path(self.mlp_channels(self.norm2(x)))
        return x



