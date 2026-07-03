# 原版
# generate_trajectory.py
# 采样轨迹（其他与原版无恙）
# 用mems=[]来采样轨迹
# save the weights of func

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import sys
import copy
import numpy as np
# from termcolor import colored
import os
import time

sys.path.append('../../')

from lib.optimizations import weight_norm, VariationalDropout, VariationalHidDropout, VariationalAttnDropout
from lib.solvers import anderson
from lib.jacobian import jac_loss_estimate, power_method

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.positional_embedding import PositionalEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits

from models.deq_transformer_CD import ConsistencyFunction, WeightSharePositionwiseFF, WeightShareSelfAttention, RelPartialLearnableDecoderLayer

class DEQTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, eval_n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weights=True, d_embed=None, div_val=1,
                 tie_projs=[False], pre_lnorm=False, wnorm=False, tgt_len=None,
                 mem_len=None, local_size=0, pretrain_steps=1, cutoffs=[], load='',
                 f_solver=anderson, b_solver=None, stop_mode="rel", logging=None):
        super().__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)  # pre-trained weight
        self.iodrop = VariationalDropout()
        self.dropout = dropout
        self.pos_drop = VariationalHidDropout(dropout=dropout)
        self.pretrain_steps = pretrain_steps

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.local_size = local_size
        self.max_klen = tgt_len + mem_len

        self.n_layer = n_layer
        self.eval_n_layer = eval_n_layer
        self.inject_conv = nn.Conv1d(d_model, 3 * d_model, kernel_size=1)  # pre-trained weight
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.func = RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout=dropout, dropatt=dropatt,
                                                    pre_lnorm=pre_lnorm, local_size=local_size)

        self.f_solver = f_solver
        self.b_solver = b_solver if b_solver else self.f_solver
        self.hook = None
        self.stop_mode = stop_mode
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        self.logging = logging or print
        if wnorm: self.func.wnorm()

        # use adaptive softmax (including standard softmax)
        # (Note: To use sample softmax, refer to the Transformer-XL implementation)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        if tie_weights:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and div_val == 1 and d_model != d_embed:
                    self.crit.out_projs[i].weight.data = self.word_emb.emb_projs[0].weight.data
                elif tie_proj and div_val != 1:
                    self.crit.out_projs[i].weight.data = self.word_emb.emb_projs[i].weight.data

        # load all param from "pretrained_wt103_deqtrans_v3.pkl"
        if len(load) > 0:
            params_dict = torch.load(load)  # params_dict = torch.load("./models/pretrained_deq.pth")
            self.load_weights(params_dict)  #
            self.logging(f"Finished loading. d_embed={self.inject_conv.weight.data.size(1)}")

        # Consistency Distrillation
        self.CD = ConsistencyFunction(n_head=n_head, d_model=d_model, d_head=d_head, d_inner=d_inner,
                                 dropout=dropout, n_layer=n_layer)
        self._cm_state_cache = {}

        # save and load the weights of func
        # self.save_func_weights("./models/")

        # load = "./models/pretrained_deq_func.pth"
        # params_dict = torch.load(load)
        # self.func.load_state_dict(params_dict)


    def reset_length(self, tgt_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len

    def load_weights(self, params_dict):
        self.load_state_dict(params_dict)

    def save_weights(self, path, name='pretrained_deq'):
        with open(os.path.join(path, f'{name}.pth'), 'wb') as f:
            self.logging(f"Saving weight state dict at {name}.pth")
            torch.save(self.state_dict(), f)

    def save_func_weights(self, path, name='pretrained_deq_func'):  # self.save_func_weights("./models/")
        with open(os.path.join(path, f'{name}.pth'), 'wb') as f:
            self.logging(f"Saving weight state dict at {name}.pth")
            torch.save(self.func.state_dict(), f)

    def load_cm_weights(self, path, device):
        key = (path, str(device))
        if key not in self._cm_state_cache:
            state = torch.load(path, map_location=device)
            if not isinstance(state, dict) or state.get('cm_time_convention') != 'z0_eps_to_fixed_T_v1':
                raise ValueError(
                    f"CM weights {path} do not match z0_eps_to_fixed_T_v1. "
                    "Retrain the CM before running -CM inference."
                )
            self._cm_state_cache[key] = state['model']
        self.CD.load_state_dict(self._cm_state_cache[key])

    def init_mems(self):
        if self.mem_len <= 0:
            self.logging("init_mems: Hmmmm... you shouldn't be here.")
            return None

        # mems is not None
        with torch.no_grad():
            mems = [torch.empty(0), torch.empty(0)]
            return mems  # For z0 and u0

    def _update_mems(self, z1s, us, z0, qlen, mlen):
        # does not deal with None
        if self.mem_len <= 0:
            self.logging("_update_mems: Hmmmm... you shouldn't be here.")
            return None

        # mems is not None
        with torch.no_grad():
            end_idx = mlen + qlen
            beg_idx = max(0, end_idx - self.mem_len)  # Account for when mlen = 0
            zs = torch.cat([z0, z1s], dim=2)
            new_z0 = zs[:, :, beg_idx:end_idx].detach().permute(2, 0, 1).contiguous()  # seq_len x bsz x d_model
            new_u0 = us[:, :, beg_idx:end_idx].detach().permute(2, 0, 1).contiguous()

            return [new_z0, new_u0]

    def _forward(self, dec_inp, mems=None, f_thres=30, b_thres=40, train_step=-1,
                 compute_jac_loss=True, spectral_radius_mode=False, writer=None, save_trajectory=False,
                 trajectory_solver='picard', CM_load=None, CM_solver='picard', CM_compare_teacher=False):
        """
        Apply the DEQ-Transformer language model on input word tokens

        :param dec_inp: Input words of shape (seq_len x bsz) and dtype torch.LongTensor
        :param mems: History madding and the transformed input corresponding to it; must be a tuple (z0, u0)
                     where z0 has dimension (bsz x d_model x pad_len) and u0 has size (bsz x 3*d_model x pad_len)
        :param f_thres: Forward pass threshold
        :param b_thres: Backward pass threshold
        :param train_step: The number of training step that the current iteration is at
        :param compute_jac_loss: Whether to return an (optional) Jacobian-stability-related loss
        :param spectral_radius_mode: Whether to estimate spectral radius at J(z*) (note: this is very slow!!)
        :param writer: Tensorboard writer
        :return: tuple(output sequence, new memory, jac loss, spec. radius)
        """
        # Assume dec_inp has shape (qlen x bsz)
        dec_inp = dec_inp.t()  # data (given)
        bsz, qlen = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        word_emb = self.iodrop(word_emb, self.dropout)
        u1s = self.inject_conv(word_emb.transpose(1, 2))  # bsz x 3*d_model x qlen

        z0, u0 = mems
        d_model = self.d_model
        if z0 is not None and z0.nelement() > 0:
            assert z0.size(2) == u0.size(2), "Padding fixed points and padding embedding dimensions don't agree"
        else:
            z0, u0 = torch.zeros(bsz, d_model, 0).to(dec_inp.device), torch.zeros(bsz, 3 * d_model, 0).to(dec_inp.device)
        mlen = z0.size(2)
        klen = mlen + qlen  # qlen is seq_len, mlen is pad_len

        pos_seq = torch.arange(klen - 1, -1, -1.0).to(dec_inp.device)
        pos_emb = self.pos_drop(self.pos_emb(pos_seq))  # bsz x d_model x (qlen + mlen) for positional embedding
        us = torch.cat([u0, u1s], dim=2)  # 2 * 2100 * 150
        z1s = torch.zeros(bsz, d_model, qlen)  # bsz x d_model x (qlen + mlen) for initial estimate of output
        func_args = [us, z0, pos_emb]
        jac_loss = torch.tensor(0.0).to(z1s)
        sradius = torch.zeros(bsz, 1).to(z1s)
        # deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)  # Ture
        deq_mode = True

        if not deq_mode:  # 显式模式
            n_layer = self.n_layer if self.training or train_step > 0 else self.eval_n_layer
            for i in range(n_layer):
                z1s = self.func(z1s, *func_args)
            new_z1s = z1s

        else:  # DEQ or CM
            # Compute the equilibrium via DEQ. When in training mode, we need to register the analytical backward
            # pass according to the Theorem 1 in the paper.
            with torch.no_grad():
                # input: func_args=【mems=[us, z0], pos_emb】, z1s=0
                # 有效的input仅为mems=[us, z0]
                # output: result['result'], result['X_list']
                time_start = time.time()
                run_teacher_solver = save_trajectory or CM_load is None or CM_compare_teacher
                use_picard_teacher = trajectory_solver == 'picard' and (save_trajectory or CM_compare_teacher)
                initial_z1s = z1s.clone().detach()
                X_list = []
                if run_teacher_solver and use_picard_teacher:
                    X_list = [initial_z1s]
                    for _ in range(f_thres):
                        z1s = self.func(z1s, *func_args)
                        X_list.append(z1s.clone().detach())
                    new_z1s = z1s
                elif run_teacher_solver:
                    result = self.f_solver(lambda z: self.func(z, *func_args), z1s, threshold=f_thres,
                                           stop_mode=self.stop_mode)
                    # print(f"Time of anderson_solver: {time.time() - time_start}")
                    new_z1s = result['result']  # torch.Size([16, 700, 150])
                    X_list = [initial_z1s] + result.get('X_list', [])
                    if not torch.equal(X_list[-1], new_z1s):
                        X_list.append(new_z1s.clone().detach())
                else:
                    new_z1s = z1s
                if save_trajectory:
                    x_traj = torch.stack(X_list, dim=0)  # z=0 -> fixed-point approximation
                    T, EPSILON = 5, 0.002
                    if len(X_list) == 1:
                        t_traj = torch.tensor([EPSILON], dtype=x_traj.dtype, device=x_traj.device)
                    else:
                        t_traj = torch.tensor([
                            (EPSILON ** (1 / 7) + (j / (len(X_list) - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7
                            for j in range(len(X_list))
                        ], dtype=x_traj.dtype, device=x_traj.device)
                    # Save x_traj_stqueezed, us, z0,
                    # posputting as a list and name it trajectory
                    trajectory = {
                        'x_traj': x_traj,  # x_traj.shape = torch.Size([39, 2, 700, 150])
                        't_traj': t_traj,  # EPSILON -> T, aligned with x_traj
                        'func_args': func_args,  # func_args = [us, z0, pos_emb]
                        'trajectory_solver': trajectory_solver,
                        'trajectory_format': 'z0_eps_to_fixed_T_v1',
                    }

                if CM_load is not None:  # CM_load = 'best_CM_model_.pth'
                    t = torch.zeros(1, 1, device=dec_inp.device).expand(bsz, 1)
                    self.CD.set_solver(CM_solver)
                    self.load_cm_weights(CM_load, dec_inp.device)

                    time_start = time.time()
                    z1s_step = z1s.unsqueeze(1)
                    z1s = self.CD(z1s_step, z1s_step, t, func_args=func_args).squeeze(1)
                    # print(f"Time of CM_solver: {time.time() - time_start}")

                    # n_layer = 15
                    # for i in range(n_layer):
                    #     z1s = self.func(z1s, *func_args)
                    # z1s = self.f_solver(lambda z: self.func(z, *func_args), z1s, threshold=f_thres,
                    #                     stop_mode=self.stop_mode)['result']

                    if CM_compare_teacher:
                        rel_diff = (z1s - new_z1s).norm() / new_z1s.norm()
                        print(f"Relative error: {rel_diff.item()}")  # 0.2233

                    new_z1s = z1s

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    z1s.requires_grad_()
                    new_z1s = self.func(z1s, *func_args)
                _, sradius = power_method(new_z1s, z1s, n_iters=150)

            if self.training:
                z1s.requires_grad_()
                new_z1s = self.func(z1s, *func_args)
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1s, z1s, vecs=1)

                def backward_hook(grad):
                    if self.hook is not None:
                        # To avoid infinite loop
                        self.hook.remove()
                        torch.cuda.synchronize()
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1s, z1s, y, retain_graph=True)[0] + grad,
                                             torch.zeros_like(grad), threshold=b_thres)['result']
                    return new_grad
                self.hook = new_z1s.register_hook(backward_hook)

        core_out = self.iodrop(new_z1s, self.dropout).permute(2, 0, 1).contiguous()  # qlen x bsz x d_model
        new_mems = self._update_mems(new_z1s, us, z0, mlen, qlen)
        trajectory = None if not save_trajectory else trajectory
        return core_out, new_mems, jac_loss.view(-1, 1), sradius.view(-1, 1), trajectory

    def forward(self, data, target, mems, train_step=-1, **kwargs):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()
        else:
            for i in range(len(mems)):
                mems[i] = mems[i].permute(1, 2, 0).contiguous()  # bsz x [-1] x seq_len
        qlen, bsz = data.size()  # 150, 16
        mlen = 0 if mems[0].nelement() == 0 else mems[0].size(2)
        klen = mlen + qlen

        # Reset dropout in self.func
        self.pos_drop.reset_mask(1, self.d_model, klen)
        self.func.reset(bsz, qlen, klen)

        tgt_len = target.size(0)
        f_thres = kwargs.get('f_thres', 30)
        b_thres = kwargs.get('b_thres', 40)
        compute_jac_loss = kwargs.get('compute_jac_loss', True)
        sradius_mode = kwargs.get('spectral_radius_mode', False)
        writer = kwargs.get('writer', None)
        save_trajectory = kwargs.get('save_trajectory', False)
        trajectory_solver = kwargs.get('trajectory_solver', 'picard')
        CM_load = kwargs.get('CM_load', None)
        CM_solver = kwargs.get('CM_solver', trajectory_solver)
        CM_compare_teacher = kwargs.get('CM_compare_teacher', False)
        hidden, new_mems, jac_loss, sradius, trajectory = self._forward(data, mems=mems, f_thres=f_thres, b_thres=b_thres,
                                                            train_step=train_step,
                                                            compute_jac_loss=compute_jac_loss,
                                                            spectral_radius_mode=sradius_mode,
                                                            writer=writer,
                                                            save_trajectory=save_trajectory,
                                                            trajectory_solver=trajectory_solver,
                                                            CM_load=CM_load,
                                                            CM_solver=CM_solver,
                                                            CM_compare_teacher=CM_compare_teacher,
                                                            )
        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1))
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss, jac_loss, sradius] + [trajectory]
        else:
            return [loss, jac_loss, sradius] + [trajectory] + new_mems
