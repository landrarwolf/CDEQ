# CM的推理
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import sys
import copy
import numpy as np
# from termcolor import colored
import os

sys.path.append('../../')

from lib.optimizations import weight_norm, VariationalDropout, VariationalHidDropout, VariationalAttnDropout
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.positional_embedding import PositionalEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits


class WeightSharePositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(WeightSharePositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.ff1_net = nn.Linear(d_model, d_inner)
        self.drop1 = VariationalHidDropout(dropout=dropout, length_first=True)
        self.ff2_net = nn.Linear(d_inner, d_model)
        self.drop2 = VariationalHidDropout(dropout=dropout, length_first=True)

        self.pre_lnorm = pre_lnorm

    def wnorm(self):
        self.ff1_net, self.ff1_fn = weight_norm(module=self.ff1_net, names=['weight'], dim=0)
        self.ff2_net, self.ff2_fn = weight_norm(module=self.ff2_net, names=['weight'], dim=0)

    def reset(self, bsz, qlen):
        self.drop1.reset_mask(bsz, self.d_inner, qlen)
        self.drop2.reset_mask(bsz, self.d_model, qlen)
        if 'ff1_fn' in self.__dict__:
            self.ff1_fn.reset(self.ff1_net)
        if 'ff2_fn' in self.__dict__:
            self.ff2_fn.reset(self.ff2_net)

    def forward(self, inp, attn_out=None):
        assert inp.size(1) == self.d_model, "Feature dimension not match!!"

        inp = inp.transpose(1, 2)
        if self.pre_lnorm:
            inp = F.layer_norm(inp, (self.d_model,))
        relu_out1 = self.drop1(F.relu(self.ff1_net(inp)))
        out2 = self.drop2(self.ff2_net(relu_out1))
        output = out2 + inp
        if not self.pre_lnorm:
            output = F.layer_norm(output, (self.d_model,))
        return output.transpose(1, 2)


class WeightShareSelfAttention(nn.Module):
    # This is similar to the RelPartialLearnableMultiHeadAttn class in Transformer-XL
    def __init__(self, d_model, n_head, d_head, dropout, dropatt,
                 pre_lnorm=False, local_size=None):
        super(WeightShareSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Conv1d(d_model, 3 * n_head * d_head, kernel_size=1, bias=False)
        self.r_net = nn.Conv1d(d_model, n_head * d_head, kernel_size=1, bias=False)
        self.r_w_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        self.r_r_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        self.o_net = nn.Conv1d(n_head * d_head, d_model, kernel_size=1)
        self.dropatt = VariationalAttnDropout(dropout=dropatt)
        self.drop = VariationalHidDropout(dropout=dropout)

        self.pre_lnorm = pre_lnorm
        self.local_size = local_size

    def wnorm(self):
        self.qkv_net, self.qkv_fn = weight_norm(module=self.qkv_net, names=['weight'], dim=0)
        self.r_net, self.r_fn = weight_norm(module=self.r_net, names=['weight'], dim=0)
        self.o_net, self.o_fn = weight_norm(module=self.o_net, names=['weight'], dim=0)

    def reset(self, bsz, qlen, klen):
        self.dropatt.reset_mask(bsz, self.n_head, qlen, klen)
        self.drop.reset_mask(bsz, self.d_model, qlen)
        if 'qkv_fn' in self.__dict__:
            self.qkv_fn.reset(self.qkv_net)
        if 'r_fn' in self.__dict__:
            self.r_fn.reset(self.r_net)
        if 'o_fn' in self.__dict__:
            self.o_fn.reset(self.o_net)

    def _rel_shift(self, x):
        # x has dimension (bsz x n_head x qlen x klen)
        bsz, n_head, qlen, klen = x.size()
        x_padded = F.pad(x, (1, 0))
        x_padded = x_padded.view(bsz, n_head, klen + 1, qlen)
        return x_padded[:, :, 1:].view_as(x)

    def forward(self, z1ss, pos_emb, u1ss, mems=None):
        # Note: In this context, qlen means the length of the sequence; and mlen describes
        #       the length of the padding. Their sum is klen.
        bsz, d_model, qlen = z1ss.size()
        r_w_bias, r_r_bias = self.r_w_bias, self.r_r_bias
        n_head, d_head = self.n_head, self.d_head
        rlen = pos_emb.size(2)

        if mems is None:
            mems = torch.tensor([]).view(0, 0, 0)
        mems = mems.to(z1ss.device)
        mlen = mems.size(2)
        cat = torch.cat([mems, z1ss], dim=-1)

        if self.pre_lnorm:
            cat = F.layer_norm(cat.transpose(1, 2), (d_model,)).transpose(1, 2)
        w_heads = self.qkv_net(cat)  # (N x 3*d_model x seq_len)
        r_head_k = self.r_net(pos_emb)

        # Input injection
        w_heads += u1ss
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=1)
        w_head_q = w_head_q[:, :, -qlen:]

        klen = w_head_k.size(2)

        w_head_q = w_head_q.view(bsz, n_head, d_head, qlen)  # bsz x n_head x d_head x qlen
        w_head_k = w_head_k.view(bsz, n_head, d_head, klen)  # bsz x n_head x d_head x klen
        w_head_v = w_head_v.view(bsz, n_head, d_head, klen)  # bsz x n_head x d_head x klen

        r_head_k = r_head_k.view(n_head, d_head, rlen)  # n_head x d_head x rlen

        # Compute attention score
        rw_head_q = w_head_q + r_w_bias[:, :, None]  # bsz x n_head x d_head x qlen
        AC = torch.einsum('bndi,bndj->bnij', rw_head_q, w_head_k)
        rr_head_q = w_head_q + r_r_bias[:, :, None]
        BD = torch.einsum('bndi,ndj->bnij', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)  # for relative positional embedding

        attn_score = AC + BD  # bsz x n_head x qlen x klen
        attn_score.mul_(self.scale)

        # Compute attention probability
        # We apply a local mask, with local horizon size of mlen
        local_size = self.local_size or 1000
        attn_mask = (torch.triu(torch.ones(qlen, klen), diagonal=1 + mlen) > 0)[None]
        attn_mask += (torch.tril(torch.ones(qlen, klen), diagonal=mlen - local_size) > 0)[None]
        attn_mask = attn_mask.to(attn_score.device)
        if attn_mask is not None and attn_mask.any().item():
            attn_score = attn_score.float().masked_fill(attn_mask[None], -float('inf')).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=-1)  # bsz x n_head x qlen x klen
        attn_prob = self.dropatt(attn_prob)

        # Compute attention vector
        attn_vec = torch.einsum('bnij,bndj->bndi', (attn_prob, w_head_v))

        # [bsz x d x qlen]
        attn_vec = attn_vec.contiguous().view(bsz, n_head * d_head, attn_vec.size(-1))

        # Linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        # Residual connection + layer normolization (if applicable)
        if self.pre_lnorm:
            out = attn_out + z1ss
        else:
            out = F.layer_norm((attn_out + z1ss).transpose(1, 2), (d_model,)).transpose(1, 2)
        return out


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        pre_lnorm = kwargs.get('pre_lnorm')
        local_size = kwargs.get('local_size', None)
        dropatt = kwargs.get('dropatt', 0.0)
        self.dec_attn = WeightShareSelfAttention(d_model, n_head, d_head, dropout=dropout, dropatt=dropatt,
                                                 pre_lnorm=pre_lnorm, local_size=local_size)
        self.pos_ff = WeightSharePositionwiseFF(d_model, d_inner, dropout, pre_lnorm=pre_lnorm)

    def wnorm(self):
        self.dec_attn.wnorm()
        self.pos_ff.wnorm()

    def reset(self, bsz, qlen, klen):
        # Reset the dropout mask(s) and re-compute the weight normalized weights at the START of each iterations
        self.dec_attn.reset(bsz, qlen, klen)
        self.pos_ff.reset(bsz, qlen)

    def forward(self, z1ss, uss, z0, *args):
        pos_emb = args[0]
        output = self.dec_attn(z1ss, pos_emb, uss, mems=z0)
        output = self.pos_ff(output)
        return output


# class ConsistencyFunction(nn.Module):
#     def __init__(self, n_head, d_model, d_head, d_inner, dropout, n_layer, func_args=None):
#         super().__init__()
#         # self.func_args = func_args
#         if func_args is not None:
#             self.update_func_args(func_args)
#
#         self.d_model = d_model
#         # self.embedding = nn.Linear(d_model + 1, d_model)  # 701 -> 700
#
#         self.embedding = nn.Sequential(
#             nn.Linear(d_model + 1, d_model*3),
#             nn.ReLU(),
#             # nn.Linear(d_model*3, d_model*3),
#             # nn.ReLU(),
#             # nn.Linear(d_model*3, d_model*3),
#             # nn.ReLU(),
#             nn.Linear(d_model*3, d_model),
#         )
#
#         self.func = RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout=dropout)
#         self.pos_drop = VariationalHidDropout(dropout=dropout)
#         wnorm = True
#         if wnorm: self.func.wnorm()
#         self.n_layer = n_layer
#
#     def update_func_args(self, func_args):
#         """Update func_args by directly assigning the new value"""
#         self.func_args = func_args
#
#
#     def forward(self, z1s: torch.Tensor, t: torch.Tensor, func_args=None):
#         if hasattr(self, 'func_args'):
#             func_args = self.func_args
#         elif not func_args:
#             raise ValueError("func_args is needed in forward function")
#         func_args = [arg.to(z1s.device) for arg in func_args]
#         bsz, n_steps, qlen = z1s.shape[0], z1s.shape[1], z1s.shape[3]
#
#         if len(func_args[2].shape) > 3:  # 如果 pos_emb 有额外维度
#             func_args[2] = func_args[2][0]  # 取第一个样本作为原始 pos_emb
#
#         ## input = self.func(0)
#         inputs = torch.zeros_like(z1s).to(z1s.device)
#         klen = func_args[2].size(2)
#
#         # Reset dropout in self.func
#         self.pos_drop.reset_mask(bsz, self.d_model, klen)
#         self.func.reset(bsz, qlen, klen)
#
#         for b in range(n_steps):  # Number of trajectory points
#             inputs[:,b] = self.func(z1s[:,b], *func_args)  # torch.Size([bsz=4, n_steps=19, 700, 150])
#             # inputs[:,b] = anderson(lambda z: self.func(z, *func_args), z1s[:, b], threshold=30)['result']  # 改anderson
#
#         ## output = MLP(input, t)
#         # 将z1s与t拼接在一起 d_model + 1
#         input_t = torch.cat([inputs, t.view(bsz, n_steps, 1, 1).tile(1, 1, 1, qlen)],
#                           dim=-2)  # bsz=4 * n_steps=19 x d_model+1=701 x qlen=150
#         # 将z1s_t的维度转换，应用嵌入，然后还原维度d_model
#         # 将z1s_t的维度从 batch_size x bsz x d_model+1 x qlen 转换为 batch_size x bsz x qlen x d_model+1
#         # torch.Size([15, 16, 700, 150])
#         outputs = self.embedding(input_t.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()
#
#         T = 5
#         EPSILON = 0.002
#
#         result = (
#                 ((T - t) / (T - EPSILON)).view(bsz, n_steps, 1, 1) * z1s  # 原为inputs
#                 +
#                 ((t - EPSILON) / (T - EPSILON)).view(bsz, n_steps, 1, 1) * outputs
#                 )
#
#         return result


def anderson_step(z_curr, z_prev, f_curr, f_prev, beta=1.0):
    """
    Achieve Anderson Acceleration for two-step results
    z_curr, z_prev: Input for the current and previous step Tensor [bsz, d_model, qlen]
    f_curr, f_prev: Output Tensor [bsz, d_model, qlen]
    """
    # 1. Calculate the residuals r(z) = f(z) - z
    r_curr = f_curr - z_curr
    r_prev = f_prev - z_prev

    # 2. Solve the weight of alpha
    # Goal: Minimize ||(1-a1)*r_prev + a1*r_curr||^2
    # Let dr = r_curr - r_prev, Minimize ||r_prev + a1*dr||^2
    dr = r_curr - r_prev

    # After the BSZ dimension, all flattened, the point product and norm are calculated
    # dot: <r_prev, dr>, norm_sq: ||dr||^2
    # The resulting shape is [bsz, 1, 1] for subsequent broadcasts
    dot_product = torch.sum(r_prev * dr, dim=(-2, -1), keepdim=True)
    norm_sq = torch.sum(dr * dr, dim=(-2, -1), keepdim=True)

    # Calculate alpha_1 (corresponding to the weight of the most recent step)
    alpha_1 = - dot_product / (norm_sq + 1e-9)
    alpha_0 = 1.0 - alpha_1

    # 3. Calculate the accelerated result (AA_update step)
    # z_next = beta * (a0*f_prev + a1*f_curr) + (1-beta) * (a0*z_prev + a1*z_curr)
    weighted_f = alpha_0 * f_prev + alpha_1 * f_curr
    weighted_z = alpha_0 * z_prev + alpha_1 * z_curr

    res = beta * weighted_f + (1 - beta) * weighted_z
    return res


class ConsistencyFunction(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, n_layer, func_args=None):
        super().__init__()
        if func_args is not None:
            self.update_func_args(func_args)

        self.d_model = d_model
        self.embedding = nn.Sequential(
            nn.Linear(d_model + 1, d_model * 3),
            nn.ReLU(),
            nn.Linear(d_model * 3, d_model),
        )

        self.func = RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout=dropout)
        self.pos_drop = VariationalHidDropout(dropout=dropout)

        wnorm = True
        if wnorm: self.func.wnorm()
        self.n_layer = n_layer

    def update_func_args(self, func_args):
        self.func_args = func_args

    def forward(self, z1s: torch.Tensor, z1s_1=None, t=None, func_args=None):
        """
        z1s: Current step iteration results [bsz, n_steps, d_model, qlen]
        z1s_1: Previous iteration results [bsz, n_steps, d_model, qlen].
        ponytail: accepts the old forward(z, t, func_args) call until callers are fully cleaned up.
        """
        if func_args is None and isinstance(t, (list, tuple)):
            func_args = t
            t = z1s_1
            z1s_1 = None
        elif t is None:
            t = z1s_1
            z1s_1 = None
        if z1s_1 is None:
            z1s_1 = z1s

        if hasattr(self, 'func_args'):
            func_args = self.func_args
        elif not func_args:
            raise ValueError("func_args is needed in forward function")

        func_args = [arg.to(z1s.device) for arg in func_args]
        bsz, n_steps, qlen = z1s.shape[0], z1s.shape[1], z1s.shape[3]

        if len(func_args[2].shape) > 3:
            func_args[2] = func_args[2][0]

        inputs = torch.zeros_like(z1s).to(z1s.device)
        klen = func_args[2].size(2)

        self.pos_drop.reset_mask(bsz, self.d_model, klen)
        self.func.reset(bsz, qlen, klen)

        # Traverse trajectory points and execute Anderson Acceleration
        for b in range(n_steps):
            z_curr = z1s[:, b]  # [bsz, d_model, qlen]
            z_prev = z1s_1[:, b]  # [bsz, d_model, qlen]

            # Calculate the function values of two steps
            f_curr = self.func(z_curr, *func_args)
            f_prev = self.func(z_prev, *func_args)

            # Use AA to get the accelerated output
            inputs[:, b] = anderson_step(z_curr, z_prev, f_curr, f_prev, beta=1.0)

        # Subsequent embedding and interpolation logic
        input_t = torch.cat([inputs, t.view(bsz, n_steps, 1, 1).tile(1, 1, 1, qlen)], dim=-2)
        outputs = self.embedding(input_t.permute(0, 1, 3, 2)).permute(0, 1, 3, 2).contiguous()

        T = 5
        EPSILON = 0.002

        result = (
                ((T - t) / (T - EPSILON)).view(bsz, n_steps, 1, 1) * z1s
                +
                ((t - EPSILON) / (T - EPSILON)).view(bsz, n_steps, 1, 1) * outputs
        )

        return result
