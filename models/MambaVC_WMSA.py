from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN
)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import pdb
from datetime import datetime
import random
import time
from wtconv import WTConv2d

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
from functools import partial
from typing import Optional, Callable, Any
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath, to_2tuple

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import numpy as np
import math

global_var = 0
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
        module,
        buffer_name,
        state_dict_key,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
        module,
        module_name,
        buffer_names,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class WMSA_V1(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA_V1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        self.SRU = SRU(input_dim) # 空间重构
        self.CRU = CRU(input_dim) # 通道重构

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        x = Rearrange('b c h w -> b h w c')(x)
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        q = rearrange(q, 'h b w p c -> b w p (h c)')
        q = rearrange(q, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        q = Rearrange('b h w c -> b c h w')(q)
        q = self.SRU(q)
        # print('q的维度:', q.shape)
        q = Rearrange('b c h w -> b h w c')(q)
        q = rearrange(q, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        q = rearrange(q, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        q = rearrange(q, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim)

        k = rearrange(k, 'h b w p c -> b w p (h c)')
        k = rearrange(k, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        k = Rearrange('b h w c -> b c h w')(k)
        k = self.CRU(k)
        # print('k的维度:', k.shape)
        k = Rearrange('b c h w -> b h w c')(k)
        k = rearrange(k, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        k = rearrange(k, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        k = rearrange(k, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim)

        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        output = Rearrange('b h w c -> b c h w')(output)
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        # pdb.set_trace()
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        # pdb.set_trace()
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col + 1, padding_row, padding_row + 1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x = self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col - 1, -padding_row, -padding_row - 1))
        return trans_x


try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)
try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)
try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
        dt_low_rank=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    if (not dt_low_rank):
        x_dbl = F.conv1d(x.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                         bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, -1, L), [D, 4 * N, 4 * N], dim=1)
        xs = CrossScan.apply(x)
        dts = CrossScan.apply(dts)
    elif no_einsum:
        xs = CrossScan.apply(x)
        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                         bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
    else:
        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        if forward_type.startswith("v0"):
            self.__initv0__(d_model, d_state, ssm_ratio, dt_rank, dropout, seq=("seq" in forward_type))
            return

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                       CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        if forward_type.startswith("debug"):
            from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations, \
                cross_selective_scanv2
            FORWARD_TYPES.update(dict(
                debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
                debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
                debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16,
                                                        self),
                debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs,
                                                          self),
                debugforward_core_mambassm_fusecscm=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
                debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm,
                                                          self),
                debugforward_core_sscore_fusecscm_fwdnrow=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fwdnrow, self),
                debugforward_core_sscore_fusecscm_bwdnrow=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_bwdnrow, self),
                debugforward_core_sscore_fusecscm_fbnrow=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fbnrow, self),
                debugforward_core_ssoflex_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm,
                                                           self),
                debugforward_core_ssoflex_fusecscm_i16o32=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
                debugscan_sharessm=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                                           cross_selective_scan=cross_selective_scanv2),
            ))
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

        if forward_type.startswith("xv"):
            self.d_state = d_state
            self.dt_rank = dt_rank
            self.d_inner = d_inner

            if d_conv > 1:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            self.act: nn.Module = act_layer()
            self.out_act: nn.Module = nn.Identity()
            del self.x_proj_weight

            if forward_type.startswith("xv1"):
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = self.forwardxv

            if forward_type.startswith("xv2"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight

            if forward_type.startswith("xv3"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv4"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            if forward_type.startswith("xv5"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight
                self.out_act = nn.GELU()

    # only used to run previous version
    def __initv0__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")

        return cross_selective_scan(
            x, x_proj_weight, None, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, delta_softplus=True,
            out_norm=out_norm,
            out_norm_shape=out_norm_shape,
            **kwargs,
        )

    def forwardv0(self, x: torch.Tensor, SelectiveScan=SelectiveScanMamba, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        y = self.forward_core(x)

        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def forwardxv(self, x: torch.Tensor, mode="xv1", **kwargs):
        B, H, W, C = x.shape
        L = H * W
        K = 4
        dt_projs_weight = getattr(self, "dt_projs_weight", None)
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        x = x.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x = self.conv2d(x)  # (b, d, h, w)
            x = self.act(x)
        x = self.in_proj(x)

        if mode in ["xv1"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            dts = CrossScanTriton.apply(dts)
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None,
                           groups=K).contiguous().view(B, -1, L)
        elif mode in ["xv2"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            dts = CrossScanTriton.apply(dts).contiguous().view(B, -1, L)
        elif mode in ["xv3"]:
            us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            dts = CrossScanTriton1b1.apply(dts.contiguous().view(B, K, -1, H, W))
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None,
                           groups=K).contiguous().view(B, -1, L)

        us = CrossScanTriton.apply(us.contiguous()).view(B, -1, L)
        Bs, Cs = Bs.view(B, K, -1, L).contiguous(), Cs.view(B, K, -1, L).contiguous()

        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)

        y: torch.Tensor = CrossMergeTriton.apply(ys)

        if out_norm_shape in ["v1"]:  # (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
        else:  # (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
            y = out_norm(y).view(B, H, W, -1)

        y = (y.to(x.dtype) if to_dtype else y)
        out = self.dropout(self.out_proj(self.out_act(y)))
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        # print(drop_path)
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
        )

        self.drop_path = DropPath(drop_path)

    def _forward(self, input: torch.Tensor):
        if self.post_norm:
            x = input + self.drop_path(self.norm(self.op(input)))
        else:
            # pdb.set_trace()
            x = input + self.drop_path(self.op(self.norm(input)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            x = input.permute(0, 2, 3, 1)
            x = self._forward(x).permute(0, 3, 1, 2)
            return x

class Fuse_Block(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            head_dim,
            window_size,
            type='W',
            # =============================
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.WMSA_V1 = WMSA_V1(self.input_dim, self.output_dim, self.head_dim, self.window_size, self.type)
        self.VSS = VSSBlock(hidden_dim=hidden_dim, drop_path=self.drop_path, use_checkpoint=use_checkpoint,
                                 norm_layer=norm_layer, ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank,
                                 ssm_conv=ssm_conv, ssm_conv_bias=ssm_conv_bias, ssm_drop_rate=ssm_drop_rate,
                                 ssm_init=ssm_init, forward_type=forward_type)
        # self.local_fea = local_feature(self.input_dim, self.output_dim)
        self.conv1 = nn.Conv2d(self.output_dim, self.output_dim*2, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.output_dim*2, self.output_dim, 3, 1, 1)
        self.gate1 = nn.Sigmoid()
        self.gate2 = nn.Sigmoid()
        # self.gate3 = nn.Sigmoid()
        self.GAP1 = nn.AdaptiveAvgPool2d(1)
        self.GAP2 = nn.AdaptiveAvgPool2d(1)
        # self.GAP3 = nn.AdaptiveAvgPool2d(1)
        # self.add1 = nn.Softmax(dim=1)
        # self.add2 = nn.Softmax(dim=1)
        # self.add3 = nn.Softmax(dim=1)
        # self.SE1 = nn.Conv2d(self.output_dim, self.output_dim, 1, 1, 0)
        # self.SE2 = nn.Conv2d(self.output_dim, self.output_dim, 1, 1, 0)
        # self.SE3 = nn.Conv2d(self.output_dim, self.output_dim, 1, 1, 0)

    def forward(self, x):
        fea = x
        WMSA, VSS = torch.split(self.conv1(x), (self.input_dim, self.input_dim), dim=1)

        WMSA_out = self.WMSA_V1(WMSA)
        feature1 = WMSA_out
        WMSA_out = self.GAP1(WMSA_out)
        # WMSA_out = self.SE1(WMSA_out)
        WMSA_out = self.gate1(WMSA_out)
        WMSA_out = feature1 * WMSA_out
        # WMSA_out = self.add1(WMSA_out)

        VSS_out = self.VSS(VSS)
        feature2 = VSS_out
        VSS_out = self.GAP2(VSS_out)
        # VSS_out = self.SE2(VSS_out)
        VSS_out = self.gate2(VSS_out)
        VSS_out = feature2 * VSS_out
        # VSS_out = self.add2(VSS_out)

        # local_fea_out = self.local_fea(local)
        # feature3 = local_fea_out
        # local_fea_out = self.GAP3(local_fea_out)
        # local_fea_out = self.SE3(local_fea_out)
        # local_fea_out = self.gate3(local_fea_out)
        # local_fea_out = feature3 * local_fea_out
        # local_fea_out = self.add3(local_fea_out)

        conv2_x = self.conv2(torch.cat((WMSA_out, VSS_out), dim=1))

        out = fea + conv2_x

        return out

class local_feature(nn.Module):
    def __init__(self, intput_dim, output_dim):
        super().__init__()
        self.input_dim = intput_dim
        self.output_dim = output_dim
        self.layer1_1 = WTConv2d(self.output_dim, self.output_dim, 3, 1, 3)
        self.layer1_relu = nn.LeakyReLU()
        self.layer1_2 = WTConv2d(self.output_dim, self.output_dim, 3, 1, 3)
        self.layer1_sigmoid = nn.Sigmoid()

        self.layer2_1 = WTConv2d(self.output_dim, self.output_dim, 3, 1, 3)
        self.layer2_2 = WTConv2d(self.output_dim, self.output_dim, 3, 1, 3)
        self.layer2_relu = nn.LeakyReLU()
        self.fuse = WTConv2d(self.output_dim, self.output_dim, 3, 1, 3)

    def forward(self, x):
        feature = x
        x_1 = self.layer1_1(x)
        x_1 = self.layer1_relu(x_1)
        x_1 = self.layer1_2(x_1)
        x_1 = self.layer1_relu(x_1)

        # x_2 = self.layer2_1(x)
        # x_2 = self.layer2_relu(x_2)
        # x_2 = self.layer2_2(x_2)

        # x_out = x_1 * x_2
        # x_out = self.fuse(x_out)
        x_out = feature + x_1
        return x_out

class down_sampling(nn.Module):
    def __init__(self, intput_dim, output_dim):
        super().__init__()
        self.input_dim = intput_dim
        self.output_dim = output_dim
        self.layer_down = conv(self.input_dim, self.output_dim, 3, 2)
        self.layer1_1 = WTConv2d(self.output_dim, self.output_dim, 5, 1, 3)
        self.layer1_relu = nn.LeakyReLU()
        self.layer1_2 = WTConv2d(self.output_dim, self.output_dim, 3, 1, 2)
        self.layer1_sigmoid = nn.Sigmoid()

        self.layer2_1 = WTConv2d(self.output_dim, self.output_dim, 5, 1, 3)
        self.layer2_2 = WTConv2d(self.output_dim, self.output_dim, 3, 1, 2)
        self.layer2_relu = nn.LeakyReLU()
        self.fuse = WTConv2d(self.output_dim, self.output_dim, 5, 1, 3)

    def forward(self, x):
        x_down = self.layer_down(x)

        x_1 = self.layer1_1(x_down)
        x_1 = self.layer1_relu(x_1)
        x_1 = self.layer1_2(x_1)
        x_1 = self.layer1_sigmoid(x_1)

        x_2 = self.layer2_1(x_down)
        x_2 = self.layer1_relu(x_2)
        x_2 = self.layer2_2(x_2)

        x_out = x_1 * x_2
        x_out = self.fuse(x_out)
        x_out = x_down + x_out
        return x_out

class up_sampling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_up = deconv(self.input_dim, self.output_dim, 3, 2)
        self.layer1_1 = WTConv2d(self.output_dim, self.output_dim, kernel_size=5, stride=1, wt_levels=3)
        self.layer1_relu = nn.LeakyReLU()
        self.layer1_2 = WTConv2d(self.output_dim, self.output_dim, kernel_size=3, stride=1, wt_levels=2)
        self.layer1_sigmoid = nn.Sigmoid()
        self.layer2_1 = WTConv2d(self.output_dim, self.output_dim, kernel_size=5, stride=1, wt_levels=3)
        self.layer2_relu = nn.LeakyReLU()
        self.layer2_2 = WTConv2d(self.output_dim, self.output_dim, kernel_size=3, stride=1, wt_levels=2)
        self.fuse = WTConv2d(self.output_dim, self.output_dim, kernel_size=5, stride=1, wt_levels=3)

    def forward(self,x):
        x_up = self.layer_up(x)
        x_1 = self.layer1_1(x_up)
        x_1 = self.layer1_relu(x_1)
        x_1 = self.layer1_2(x_1)
        x_1 = self.layer1_sigmoid(x_1)
        x_2 = self.layer2_1(x_up)
        x_2 = self.layer1_relu(x_2)
        x_2 = self.layer2_2(x_2)
        x_out = x_1 * x_2
        x_out = self.fuse(x_out)
        x_out = x_up + x_out

        return x_out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MAMBAVC(CompressionModel):
    def __init__(self, depths=[2, 2, 2, 2], head_dim=[8, 16, 32, 32 ,16, 8], drop_path_rate=0.1, N=128, M=320, num_slices=5, max_support_slices=5,
                 **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.depths = depths
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.m_down1 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=self.head_dim[0],
                                   window_size=self.window_size, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                   drop_path=dpr[i], use_checkpoint=False,
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(depths[0])] + [conv(N*2, N*2, kernel_size=3, stride=2)]

        self.m_down2 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=self.head_dim[1],
                                   window_size=self.window_size, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                   drop_path=dpr[i + depths[0]], use_checkpoint=False,
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(depths[1])] + [conv(N*2, N*2, kernel_size=3, stride=2)]

        self.m_down3 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=self.head_dim[2],
                                   window_size=self.window_size, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                   drop_path=dpr[i + depths[0] + depths[1]], use_checkpoint=False,
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                        for i in range(depths[2])] + [conv(N*2, M, kernel_size=3, stride=2)]

        self.g_a = nn.Sequential(
            *[conv(3, N*2, kernel_size=5, stride=2)] + self.m_down1 + self.m_down2 + self.m_down3)

        self.ha_down1 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=32,
                                   window_size=4, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                    drop_path=dpr[i + depths[0] + depths[1] + depths[2]],
                                  use_checkpoint=False,
                                  norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                                  ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                         for i in range(depths[3])] + [conv(N*2, 192, kernel_size=3, stride=2)]

        self.h_a = nn.Sequential(
            *[conv(M, N*2, kernel_size=3, stride=2)] + self.ha_down1
        )

        depths = depths[::-1]

        self.hs_up1 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=32,
                                   window_size=4, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                  drop_path=dpr[i], use_checkpoint=False,
                                norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                                ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                       for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]

        self.h_mean_s = nn.Sequential(
            *[deconv(192, N*2, kernel_size=3, stride=2)] + self.hs_up1
        )

        self.hs_up2 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=32,
                                   window_size=4, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                  drop_path=dpr[i], use_checkpoint=False,
                                norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                                ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                       for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]

        self.h_scale_s = nn.Sequential(
            *[deconv(192, N*2, kernel_size=3, stride=2)] + self.hs_up2
        )

        self.m_up1 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=self.head_dim[3],
                                   window_size=self.window_size, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                 drop_path=dpr[i + depths[0]], use_checkpoint=False,
                               norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                               ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[1])] + [deconv(N*2, N*2, kernel_size=3, stride=2)]

        self.m_up2 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=self.head_dim[4],
                                   window_size=self.window_size, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                 drop_path=dpr[i + depths[0] + depths[1]], use_checkpoint=False,
                               norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                               ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[2])] + [deconv(N*2, N*2, kernel_size=3, stride=2)]

        self.m_up3 = [Fuse_Block(input_dim=N * 2, output_dim=N*2, head_dim=self.head_dim[5],
                                   window_size=self.window_size, type='W' if not i%2 else 'SW', hidden_dim=2 * N,
                                 drop_path=dpr[i + depths[0] + depths[1] + depths[2]],
                               use_checkpoint=False,
                               norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
                               ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[3])] + [deconv(N*2, 3, kernel_size=5, stride=2)]

        self.g_s = nn.Sequential(*[deconv(M, N*2, kernel_size=3, stride=2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((M + (M // self.num_slices) * min(i, 5)), (M + (M // self.num_slices) * min(i, 5)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((M + (M // self.num_slices) * min(i, 5)), (M + (M // self.num_slices) * min(i, 5)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + (M // self.num_slices) * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (M // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + (M // self.num_slices) * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (M // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + (M // self.num_slices) * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (M // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        # pdb.set_trace()
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales},
            "y": y,
            "z": z
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # pdb.set_trace()
        super().load_state_dict(state_dict)

    #     @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        y_hat = torch.cat(y_hat_slices, dim=1)

        s = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        e = time.time()
        Time = e - s
        return {"x_hat": x_hat,
                "Time": Time}

    def compress_decompress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        strings = [y_strings, z_strings]
        shape = z.size()[-2:]

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        y_hat = torch.cat(y_hat_slices, dim=1)

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fuse_Block = Fuse_Block(256, 256, 8, 8, 'W', hidden_dim=256, drop_path=0, use_checkpoint=False,
    #                              norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
    #                              ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2").to(device)
    MAMBAVC = MAMBAVC().to(device)
    # down_sampling = up_sampling(256, 256)
    input_image = torch.zeros([1, 3, 256, 256]).to(device)
    # input_image = Rearrange('b c h w -> b h w c')(input_image)
    feature = MAMBAVC(input_image)
    # feature = Rearrange('b h w c -> b c h w')(feature)
    print(feature)