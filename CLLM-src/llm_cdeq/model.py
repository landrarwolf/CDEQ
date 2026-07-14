from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


CHECKPOINT_SCHEMA = "llm_cdeq_checkpoint_v1"


class RMSNormNoAffine(nn.Module):
    """Stateless RMSNorm compatible with the official PyTorch 2.1 runtime."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=value.dtype)


class ResidualMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class InitialStatePredictor(nn.Module):
    def __init__(self, hidden_size: int, rank: int, multiplier: int = 3):
        super().__init__()
        self.norm = RMSNormNoAffine(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.mlp = ResidualMLP(rank, multiplier * rank, rank)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp.net[-1].weight)
        nn.init.zeros_(self.mlp.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.mlp(self.down(self.norm(state))))
        return state + delta


class CDEQAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int = 512,
        multiplier: int = 3,
        terminal: float = 5.0,
        use_initializer: bool = False,
    ):
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or terminal <= 0:
            raise ValueError("hidden_size, rank, and terminal must be positive")
        self.hidden_size = hidden_size
        self.rank = rank
        self.terminal = float(terminal)
        self.norm = RMSNormNoAffine(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.updater = ResidualMLP(rank + 1, multiplier * rank, rank)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        self.initializer = (
            InitialStatePredictor(hidden_size, rank, multiplier) if use_initializer else None
        )

    @property
    def use_initializer(self) -> bool:
        return self.initializer is not None

    def initialize(self, state: torch.Tensor) -> torch.Tensor:
        if self.initializer is None:
            return state
        return self.initializer(state)

    def consistency(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError("state must have shape [batch, block, hidden]")
        batch, block, _ = state.shape
        time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim == 1:
            if time.shape[0] != batch:
                raise ValueError("one-dimensional time must match the batch size")
            time = time[:, None].expand(batch, block)
        if time.shape != (batch, block):
            raise ValueError("time must be scalar, [batch], or [batch, block]")

        latent = self.down(self.norm(state))
        normalized_time = (time / self.terminal).clamp(0, 1).unsqueeze(-1)
        raw = self.updater(torch.cat((latent, normalized_time), dim=-1))
        # Algebraically equivalent to the CDEQ boundary mix, with an exact
        # zero residual at t=T.
        scale = (1 - normalized_time)
        return state + scale * self.up(raw - latent)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        apply_initializer: bool = False,
        detach_initializer: bool = False,
    ) -> torch.Tensor:
        if apply_initializer:
            state = self.initialize(state)
            if detach_initializer:
                state = state.detach()
        return self.consistency(state, time)

    def consistency_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("initializer."):
                yield parameter

    def initializer_parameters(self):
        if self.initializer is None:
            return iter(())
        return self.initializer.parameters()


@torch.no_grad()
def update_ema_(ema: nn.Module, online: nn.Module, decay: float) -> None:
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    online_state = online.state_dict()
    for name, ema_value in ema.state_dict().items():
        online_value = online_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(online_value, alpha=1 - decay)
        else:
            ema_value.copy_(online_value)


def make_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class AdapterMetrics:
    endpoint_relative_error: float
    token_agreement: float
    exact_block_match: float
    examples: int
