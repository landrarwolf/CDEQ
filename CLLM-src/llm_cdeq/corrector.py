from __future__ import annotations

import math

import torch
from torch import nn

from .model import InitialStatePredictor, RMSNormNoAffine


class TimeEmbedding(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, rank),
            nn.SiLU(),
            nn.Linear(rank, rank),
        )

    def forward(self, normalized_time: torch.Tensor) -> torch.Tensor:
        return self.net(normalized_time.unsqueeze(-1))


class CausalCorrectorBlock(nn.Module):
    def __init__(self, rank: int, heads: int, ffn_size: int):
        super().__init__()
        if rank % heads:
            raise ValueError("corrector rank must be divisible by attention heads")
        self.attention_norm = RMSNormNoAffine(rank)
        self.attention = nn.MultiheadAttention(
            rank, heads, dropout=0.0, batch_first=True, bias=True
        )
        self.ffn_norm = RMSNormNoAffine(rank)
        self.ffn = nn.Sequential(
            nn.Linear(rank, ffn_size),
            nn.SiLU(),
            nn.Linear(ffn_size, rank),
        )

    def forward(self, value: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
            is_causal=True,
        )
        value = value + attended
        return value + self.ffn(self.ffn_norm(value))


class TransformerResidualCorrector(nn.Module):
    """Block-aware CDEQ+ residual applied after one official CLLM step."""

    def __init__(
        self,
        hidden_size: int = 4096,
        rank: int = 512,
        *,
        block_size: int = 16,
        layers: int = 1,
        heads: int = 8,
        ffn_size: int = 2048,
        terminal: float = 5.0,
        preserve_first_position: bool = True,
    ):
        super().__init__()
        if min(hidden_size, rank, block_size, layers, heads, ffn_size) <= 0:
            raise ValueError("corrector dimensions and layer counts must be positive")
        if terminal <= 0:
            raise ValueError("terminal must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.terminal = float(terminal)
        self.preserve_first_position = bool(preserve_first_position)
        self.input_norm = RMSNormNoAffine(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.time_embedding = TimeEmbedding(rank)
        self.position_embedding = nn.Parameter(torch.empty(block_size, rank))
        self.blocks = nn.ModuleList(
            CausalCorrectorBlock(rank, heads, ffn_size) for _ in range(layers)
        )
        self.output_norm = RMSNormNoAffine(rank)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.position_embedding, mean=0.0, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(self.up.weight)
        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _time(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch, block, _ = state.shape
        value = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        if value.ndim == 0:
            value = value.expand(batch, block)
        elif value.ndim == 1:
            if value.shape[0] != batch:
                raise ValueError("one-dimensional time must match batch size")
            value = value[:, None].expand(batch, block)
        elif value.shape != (batch, block):
            raise ValueError("time must be scalar, [batch], or [batch, block]")
        return (value / self.terminal).clamp(0, 1)

    def residual(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.block_size, self.hidden_size):
            raise ValueError(
                f"state must have shape [batch, {self.block_size}, {self.hidden_size}]"
            )
        normalized_time = self._time(time, state)
        value = self.down(self.input_norm(state))
        value = value + self.position_embedding.to(dtype=value.dtype)[None]
        value = value + self.time_embedding(normalized_time)
        mask = self.causal_mask.to(device=value.device)
        for block in self.blocks:
            value = block(value, mask)
        delta = self.up(self.output_norm(value))
        if self.preserve_first_position:
            delta = torch.cat((torch.zeros_like(delta[:, :1]), delta[:, 1:]), dim=1)
        return delta

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        if disable_adapter:
            return state
        normalized_time = self._time(time, state)
        boundary = normalized_time.eq(1)
        if bool(boundary.all()):
            # Avoid arithmetic entirely: this is a bitwise identity boundary.
            return state
        corrected = state + (1 - normalized_time).unsqueeze(-1) * self.residual(state, time)
        if bool(boundary.any()):
            corrected = torch.where(boundary.unsqueeze(-1), state, corrected)
        return corrected


def build_initializer(hidden_size: int, rank: int) -> InitialStatePredictor:
    return InitialStatePredictor(hidden_size, rank, multiplier=3)


def corrector_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
