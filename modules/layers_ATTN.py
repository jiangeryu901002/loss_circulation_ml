import math
import numpy as np
import warnings
import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter, Module, attention
import torch.nn.functional as F
from torch.autograd import Function
from utils import init_network
from typing import Any, Callable, Optional, Union


class GATLayer(Module):
    def __init__(self, in_features, hidden_features, out_features, num_node, n_head, dropout=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_features
        self.hid_dim = hidden_features
        self.out_dim = out_features
        self.n = num_node
        self.heads = n_head
        self.head_dim = self.hid_dim // self.heads
        # self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.attn = nn.MultiheadAttention(self.hid_dim, self.heads, batch_first=True)#Parameter(torch.ones(self.n, self.n, self.heads))
        self.k_proj, self.q_proj, self.v_proj = nn.Linear(self.in_dim, self.hid_dim, bias=False), nn.Linear(self.in_dim,
                                                                                                            self.hid_dim,
                                                                                                            bias=False), nn.Linear(
            self.in_dim, self.hid_dim, bias=False)
        self.gates = nn.Parameter(torch.ones(num_node, num_node))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)

        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features // 2, hidden_features),
        )
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        init_network(self.modules)

    def forward(self, mask, X):
        B, N, C = X.shape
        Q, K, V = self.q_proj(X).reshape(B, N, self.heads, self.head_dim), self.k_proj(X).reshape(B, N, self.heads,
                                                                                                  self.head_dim), self.v_proj(
            X).reshape(B, N, self.heads, self.head_dim)  # (B, N, hid_dim)
        attn = torch.einsum("bqhd,bkhd->bqkh", [Q, K])
        markoff_value = -1e8
        attn = attn * (1 - mask[None, :, :, None]) + markoff_value * mask[None, :, :, None]
        attention = (attn / (self.head_dim ** (1 / 2))).clamp(-5, 5)
        attention = torch.softmax(attention * self.gates[None, :, :, None], dim=-2).float()
        # attention = torch.softmax((attn / (self.head_dim ** (1 / 2))).clamp(-5, 5), dim=1).float()
        entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1).mean()
        out = torch.einsum("bqkh,bkhd->bqhd", [attention, V]).reshape(B, N, self.heads * self.head_dim)
        # out, attention = self.attn(X, X, X, attn_mask=mask, average_attn_weights=False)
        # print('attention', attention.shape)
        if len(attention.shape) > 2:
            # attention = attention.reshape(B, self.heads, N, N)
            attention = attention.mean(axis=0)
        out = self.dropout(self.norm1(out + X))
        forward = self.act(self.feed_forward(out))
        out = self.dropout(self.norm2(forward + out))
        # out = self.dropout(out + X)
        out = self.fc_out(out)
        # print(type(attention))
        return out, (attention, entropy)


class HyperGATLayer(Module):
    def __init__(self, in_features, hidden_features, out_features, num_node, n_head, dropout=0.2):
        super(HyperGATLayer, self).__init__()
        self.in_dim = in_features
        self.hid_dim = hidden_features
        self.out_dim = out_features
        self.n = num_node
        self.heads = n_head
        self.head_dim = self.hid_dim // self.heads
        self.start_fn = nn.Linear(self.in_dim, self.hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_features, hidden_features * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features * 2, hidden_features),
        )
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        init_network(self.modules)

    def forward(self, mask, X, attn, V):
        B, N, C = X.shape

        # Split the embedding into self.heads different pieces
        # X = self.act(self.start_fn(X))  # (N, T, hid_dim)
        # values = X.reshape(B, N, self.heads, self.head_dim) @ V.T  # embed_size维拆成 heads×head_dim
        X = self.start_fn(X)  # (N, T, hid_dim)
        values = X.reshape(B, N, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim

        markoff_value = -1e8
        attn = attn * mask + markoff_value * (1 - mask)
        attention = torch.softmax((attn / (self.head_dim ** (1 / 2))).clamp(-5, 5), dim=1).float()  # 在K维做softmax，和为1
        # attention shape: (N, N, T, heads)

        out = torch.einsum("qkh,bkhd->bqhd", [attention, values]).reshape(B, N, self.heads * self.head_dim)
        # attention shape: (N, N, T, heads) values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim)

        out = self.dropout(self.norm1(out + X))
        forward = self.act(self.feed_forward(out))
        out = self.dropout(self.norm2(forward + out))
        # out = self.dropout(out + X)
        out = self.fc_out(out)

        return out


class CustomizedTransformerDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            device=None,
            dtype=None,
            use_sa_block=False,
            pct_attn: Optional[float] = None,
            thr_attn: Optional[float] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        self.multihead_attn = MultiheadFlexAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            score_percentile=pct_attn,
            score_threshold=thr_attn,
            **factory_kwargs,
        )

        self.use_sa_block = use_sa_block
        self.pct_attn = pct_attn
        self.thr_attn = thr_attn

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> torch.Tensor:

        x = tgt
        if self.norm_first:
            # x = x + self._sa_block(
            #     self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            # )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            # x = self.norm1(
            #     x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
            self,
            x: torch.Tensor,
            mem: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor],
            is_causal: bool = False,
    ) -> torch.Tensor:
        def score_thr(score, batch, head, q_idx, k_idx):
            return score if score > self.thr_attn else 1e-8

        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)


class MultiheadFlexAttention(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
            score_percentile: Optional[float] = None,
            score_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear_Q = nn.Linear(
            self.embed_dim, self.embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.linear_K = nn.Linear(
            self.kdim, self.embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.linear_V = nn.Linear(
            self.vdim, self.embed_dim, bias=bias, device=device, dtype=dtype
        )
        # for the type: ignore, see https://github.com/pytorch/pytorch/issues/58969
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias,
                                  device=device, dtype=dtype)  # type: ignore[assignment]

        self.score_percentile = score_percentile
        self.score_threshold = score_threshold

    def score_percentile_mod(self, score):  # , b, h, q_idx, kv_idx):
        p = torch.quantile(score, self.score_percentile, dim=-1, keepdim=True)
        return score < p[:, :, :, None]   # noop - standard attention

    def score_threshold_mod(self, score):  # , b, h, q_idx, kv_idx):
        return score < self.score_threshold  # noop - standard attention

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        if is_causal:
            raise AssertionError("causal mask not supported by AO MHA module")

        if self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))

        tgt_len, bsz, embed_dim_to_check = query.size()
        assert self.embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert (
                head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.linear_Q(query) * scaling
        k = self.linear_K(key)
        v = self.linear_V(value)

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for `attn_mask` in `nn.MultiheadAttention` is deprecated. "
                    "Use bool tensor instead.",
                    stacklevel=3,
                )
                attn_mask = attn_mask.to(torch.bool)
            assert (
                    attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for `key_padding_mask` in `nn.MultiheadAttention` is deprecated. "
                "Use bool tensor instead.",
                stacklevel=3,
            )
            key_padding_mask = key_padding_mask.to(torch.bool)
        if self.bias_k is not None and self.bias_v is not None:
                bias_k = self.bias_k
                assert bias_k is not None
                bias_v = self.bias_v
                assert bias_v is not None

                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k_zeros = torch.zeros((k.size(0), 1) + k.size()[2:])
            k = torch.cat([k, k_zeros], dim=1)
            v_zeros = torch.zeros((v.size(0), 1) + k.size()[2:])
            v = torch.cat([v, v_zeros], dim=1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if self.score_percentile is not None:
            sp_mask = self.score_percentile_mod(attn_output_weights)
            attn_output_weights.masked_fill_(sp_mask, float("-inf"))

        if self.score_threshold is not None:
            st_mask = self.score_threshold_mod(attn_output_weights)
            attn_output_weights = attn_output_weights * (1-st_mask.float()) + torch.nan_to_num(float('-inf')*st_mask.float(), 0)
            # attn_output_weights.masked_fill_(st_mask, float("-inf"))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        if self.batch_first:
            attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        else:
            attn_output = (
                attn_output.transpose(0, 1)
                .contiguous()
                .view(tgt_len, bsz, self.embed_dim)
            )

        attn_output = self.out_proj(attn_output)  # type: ignore[has-type]

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


class Conv(nn.Module):
    def __init__(self, kernel_size, in_channels, hidden_channels, out_channels, num_layers=0, dropout=0.2):
        super(Conv, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=0)
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.drop_in = nn.Dropout(dropout)
        self.act_in = nn.ReLU()  # nn.Tanh()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)])
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        # x = torch.cat([torch.zeros((B, C, self.kernel_size-1)).to(x.device), x], dim=-1)
        x = self.drop_in(self.act_in(self.conv_in(x)))
        for i in range(self.num_layers):
            # x = torch.cat([torch.zeros((B, x.shape[1], self.kernel_size-1)).to(x.device), x], dim=-1)
            x = self.convs[i](x) + x[..., 2:]
        # x = torch.cat([torch.zeros((B, x.shape[1], self.kernel_size-1)).to(x.device), x], dim=-1)
        x = self.conv_out(x)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=0, dropout=0.2):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.drop_in = nn.Dropout(dropout)
        self.act_in = nn.ReLU()  # nn.Tanh()
        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)])
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.drop_in(self.act_in(self.lin_in(x)))  # B, T, C
        for i in range(self.num_layers):
            x = self.lins[i](x) + x
        x = self.lin_out(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    """Implements positional encoding as described in 'Attention is All You Need'."""

    def __init__(self, d_model, dropout=0.1, max_len=512, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # Create a long enough 'pe' matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        if batch_first:
            pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        else:
            pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_length, batch_size, d_model)
        Returns:
            x: Tensor after adding positional encoding.
        """
        # print(x.shape, self.pe[:x.size(1)].shape)
        x = x + self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(1)]
        return self.dropout(x)


class GRU(nn.Module):
    def __init__(self, num_series, hidden, num_layers=3):
        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True, num_layers=num_layers)
        self.gru.flatten_parameters()
        # self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        # Initialize hidden states
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, X, z, connection, mode='train'):
        X = X[:, :, torch.where(connection != 0)[0]]
        # if mode == 'train':
        X_right, hidden_out = self.gru(X, z)
        X_right = self.linear(X_right)

        return X_right, hidden_out
