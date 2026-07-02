# Description: This file contains the implementation of the CM algorithm.
#
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from copy import deepcopy
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.deq_transformer import DEQTransformerLM

# 环境：IGNN (2)，服务器：(2)
# 设置显卡号
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2333, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


class ConsistencyFunction(nn.Module):
    def __init__(self, nfeat_in, nfeat_out, adj: torch.Tensor, nhid=None, dropout=0.5):
        super(ConsistencyFunction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.dropout = dropout
        self.adj = adj

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
            x: shape (batch, node, fea) t: shape (batch, 1); return shape (batch, node, fea)
        """
        if len(x.shape) == 3:
            node = x.shape[1]
            batch_size = x.shape[0]
            inputs = torch.cat([x, t.unsqueeze(-1).expand(batch_size, node).unsqueeze(-1)], dim=-1)

            # outputs = f(inputs)
            outputs = self.mlp(inputs)

            return (
                    ((T - t) / (T - EPSILON)).view(-1, 1, 1) * x
                    + ((t - EPSILON) / (T - EPSILON)).view(-1, 1, 1) * outputs
            )

        # len(x.shape) == 2
        # else:
        #     node = x.shape[0]
        #     batch_size = 1
        #     inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        #     outputs = F.relu(self.gc(inputs, self.adj))
        #
        #     return ((T - t) / (T - EPSILON)).unsqueeze(-1) * x + ((t - EPSILON) / (T - EPSILON)).unsqueeze(-1) * outputs


# load data
x_list = torch.load('x_list.pt')
# 将x_traj逆向排列 -> X[0]为不动点，X[-1]为初始点
x_list = x_list[::-1]
# 将list转为矩阵
x_traj = torch.stack(x_list, dim=0)

# load adj
adj = torch.load('adj.pt')

# 有n_samples=31个样本点，每个样本点的shape为3327*256
n_samples, num_nodes, nfeat = x_traj.shape[0], x_traj.shape[1], x_traj.shape[2]

# Create dataset and dataloader
dataset = TensorDataset(x_traj)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# 从数据中直接学(简称CT，Consistency Training) 直接从训练数据中估计
# Consistency Training
ct = ConsistencyFunction(nfeat, nfeat, adj).cuda()
ct_optimizer = torch.optim.AdamW(ct.parameters(), lr=5e-3)

# Define global constants
T = 5  # The maximum time, which is equal to the maximum noise standard deviation(噪声标准差)
EPSILON = 0.002  # The minimum time
N_EPOCHS = 200  # Number of epochs

ct_ema = deepcopy(ct)
params_ema = ct_ema.state_dict()

# 在CT之前，构造初值初始化器
with trange(N_EPOCHS) as pbar:
    for epoch in range(N_EPOCHS):
        tot_loss = 0.0
        n_steps = int(6 + 10 * epoch / N_EPOCHS)  # each epoch sample 12~16 steps

        t_steps = [
            (EPSILON ** (1 / 7) + (j / (n_steps - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7
            for j in range(0, n_steps)]  # 0~5 共n_steps==10个时间节点 步长先小后大
        # t_steps = [T - t_steps[j] for j in range(0, n_steps)]

        # t_steps = [
        #     (EPSILON ** 7 + (j / (n_steps - 1)) * (T ** 7 - EPSILON ** 7) ** (1 / 7))
        #     for j in range(0, n_steps)]  # 0~5 共n_steps==10个时间节点 步长先小后大


        for x, in dataloader:
            batch_size = x.shape[0]

            n_1 = [random.randint(1, n_steps - 1) for _ in range(batch_size)]
            tn_1 = torch.tensor([t_steps[i] for i in n_1]).cuda()  # 当前时间节点
            tn = torch.tensor([t_steps[i - 1] for i in n_1]).cuda()  # 前一步时间节点

            # # Sample x_{t_{n+1}} = x + t_{n+1} * z, x_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) * z = x + t_n * z
            # z = torch.randn(batch_size, 1)
            # x_tn_1 = x + tn_1.unsqueeze(-1) * z
            # x_tn = x + tn.unsqueeze(-1) * z

            # Sample x_tn_1 and x_tn
            x_tn_1 = x[n_1]
            x_tn = x[(np.array(n_1) - 1).tolist()]

            # 相邻损失：F(x_tn_1, tn_1) -> F(x_tn, tn)
            # Compute loss and update consistency function
            with torch.no_grad():
                out_tn = ct_ema(x_tn, tn)
            loss_1 = F.mse_loss(ct(x_tn_1, tn_1), out_tn)

            # 全局损失:F(x_tn, tn) -> x_fixed
            x_fixed = x_traj[0].unsqueeze(0).expand(batch_size, num_nodes, nfeat)  # 不动点
            # t_steps_0 = torch.tensor(t_steps[0]).cuda()  # t=EPSILON
            loss_2 = F.mse_loss(ct(x_tn, tn), x_fixed)
            # loss_2 = loss_2 * (1/batch_size)

            loss = 0 * loss_1 + 1 * loss_2

            loss.backward()
            ct_optimizer.step()
            ct_optimizer.zero_grad()

            params_ema = {k: 0.98 * params_ema[k] + 0.02 * v for k, v in ct.state_dict().items()}
            # k = 'gc.weight', v = value
            ct_ema.load_state_dict(params_ema)
            tot_loss += loss.item()
        pbar.set_postfix(loss=tot_loss / len(dataloader))
        pbar.update()

        # Inference
        with torch.no_grad():
            x_ini = x_traj[-1]
            # 将x_ini的shape变为(1, 3327, 256)
            # x_ini = x_ini.unsqueeze(0).cuda()

            # 一步推理
            # old
            # t = (torch.ones(num_nodes) * T).cuda()
            # x = ct(x_ini, t)

            # new
            t = torch.tensor(t_steps[-1]).cuda()  # 直接取最后时间步：T == 5
            x = ct(x_ini.unsqueeze(0), t)

            # 计算x与x_traj最后一个样本点的相对误差
            rel_diff = (x - x_traj[0]).norm() / x_traj[0].norm()
            print(f" Relative error: {rel_diff.item()}")

# # Visualize the generated distribution of consistency_training
# n_samples = 10000
# xT = torch.randn(n_samples, 1) * T
# with torch.no_grad():
#     x = ct(xT, torch.ones(n_samples) * T)
# plt.hist(x.numpy()[:, 0], bins=100, density=True)
# plt.title("Generated distribution of Consistency Training")
