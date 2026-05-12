# Description: This file contains the implementation of the CM algorithm.
#
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
# import matplotlib.pyplot as plt
from copy import deepcopy
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ConsistencyFunction(nn.Module):
    def __init__(self, nfeat_in, nfeat_out, nhid=None):
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