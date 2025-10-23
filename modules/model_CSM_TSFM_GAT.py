import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from IGNNs.layers_ATTN import GATLayer, HyperGATLayer
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout, squeeze_diag, DFT_series_decomp
import torch.sparse as sparse
from ODEs.HyperODE import FullGRUODECell_Autonomous as LatentODEfunc #LatentODEfunc
import math
from IGNNs.layers import MixerBlock, Mlp, Conv
from tsfm_public.toolkit.get_model import get_model
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

class GATConfig(PretrainedConfig):
    def __init__(self,
                 # tsp=None,
                 # adj=None,
                 nfeat=14,
                 nhid=32,
                 num_node=14,
                 num_time=12,
                 num_pred=12,
                 n_head=4,
                 dropout=0.1,
                 # device='cuda',
                 finetune=False,
                 **kwargs):
        self.nfeat, self.nhid = nfeat, nhid
        self.num_node = num_node
        self.num_time = num_time
        self.num_pred = num_pred
        self.n_head = n_head
        self.dropout = dropout
        self.finetune = finetune
        # GNN for the first prediction step
        # self.adj = adj
        # self.tsp = tsp
        # self.dd = device
        super().__init__(**kwargs)

class GAT(PreTrainedModel):
    config_class = GATConfig
    def __init__(self, config, tsp, adj):
        super().__init__(config)

        self.adj = adj
        self.tsp = tsp
        self.gat = GATLayer(config.nfeat, config.nhid, 1, config.num_node, config.n_head, config.dropout)
        self.norm_g = nn.BatchNorm1d(config.num_node)
        self.TSFM = get_model(
            "ibm-granite/granite-timeseries-ttm-r2",
            context_length=config.num_time,
            prediction_length=config.num_pred,
            num_input_channels=tsp.num_input_channels,
            decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
            prediction_channel_indices=tsp.prediction_channel_indices,
        )

        if config.finetune:
            for param in self.TSFM.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        freq_token: Optional[torch.Tensor] = None,
        static_categorical_values: Optional[torch.Tensor] = None):
        x = torch.diag_embed(past_values, dim1=-2, dim2=-1)
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)
        # causal for residual
        out_g = self.gat(self.adj, x).reshape(B, T, N)
        out_g = self.norm_g(out_g.permute(0, 2, 1)).permute(0, 2, 1)
        outs = self.TSFM(out_g, future_values, past_observed_mask, future_observed_mask,
                         output_hidden_states, return_loss, return_dict, freq_token)
        return outs