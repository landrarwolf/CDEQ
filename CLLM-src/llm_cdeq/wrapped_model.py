from __future__ import annotations

import time as wall_time
from dataclasses import dataclass

import torch
from torch import nn

from .cllm_step import CLLMPrefill, CLLMStepOutput, OfficialCLLMSingleStep
from .corrector import TransformerResidualCorrector


@dataclass(frozen=True)
class WrappedStepOutput:
    base: CLLMStepOutput
    hidden: torch.Tensor
    logits: torch.Tensor
    tokens: torch.Tensor
    cllm_backbone_nfe: int
    corrector_nfe: int
    initializer_nfe: int
    corrector_latency_seconds: float


@dataclass(frozen=True)
class WrappedRolloutOutput:
    tokens: torch.Tensor
    rounds: int
    cllm_backbone_nfe: int
    corrector_nfe: int
    initializer_nfe: int
    prompt_prefill_nfe: int
    corrector_latency_seconds: float
    converged: bool


class WrappedCLLM(nn.Module):
    """The composite operator A_theta(J_phi(x, y), t)."""

    def __init__(
        self,
        cllm_step: OfficialCLLMSingleStep,
        corrector: TransformerResidualCorrector,
        *,
        initializer: nn.Module | None = None,
    ):
        super().__init__()
        if cllm_step.hidden_size != corrector.hidden_size:
            raise ValueError("CLLM and corrector hidden sizes do not match")
        if cllm_step.block_size != corrector.block_size:
            raise ValueError("CLLM and corrector block sizes do not match")
        self.cllm_step = cllm_step
        self.corrector = corrector
        self.initializer = initializer

    @property
    def lm_head(self) -> nn.Module:
        return self.cllm_step.lm_head

    def train(self, mode: bool = True):
        super().train(mode)
        # The official operator remains frozen and deterministic even while the
        # lightweight corrector is trained.
        self.cllm_step.eval()
        return self

    def prefill(self, prompt_ids: torch.Tensor) -> CLLMPrefill:
        return self.cllm_step.prefill(prompt_ids)

    def forward(
        self,
        prefill: CLLMPrefill,
        current_tokens: torch.Tensor,
        time: torch.Tensor,
        *,
        round_index: int,
        disable_adapter: bool = False,
    ) -> WrappedStepOutput:
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        base = self.cllm_step(prefill, current_tokens)
        normalized_time = self.corrector._time(time, base.canonical_hidden)
        boundary = normalized_time.eq(1)
        if disable_adapter or bool(boundary.all()):
            hidden = base.canonical_hidden
            corrector_nfe = 0
            initializer_nfe = 0
            corrector_latency = 0.0
        else:
            hidden = base.canonical_hidden
            initializer_nfe = 0
            if self.initializer is not None and round_index == 0:
                hidden = self.initializer(hidden).detach()
                # Preserve the prompt-decided token position across every
                # official Jacobi iteration.
                hidden = torch.cat((base.canonical_hidden[:, :1], hidden[:, 1:]), dim=1)
                initializer_nfe = 1
            started = wall_time.perf_counter()
            corrector_dtype = next(self.corrector.parameters()).dtype
            hidden = self.corrector(hidden.to(dtype=corrector_dtype), time)
            if bool(boundary.any()):
                # A mixed-time batch must preserve the official CLLM hidden on
                # every terminal row/token even if other samples use Init.
                hidden = torch.where(
                    boundary.unsqueeze(-1), base.canonical_hidden, hidden
                )
            corrector_latency = wall_time.perf_counter() - started
            corrector_nfe = 1
        logits = self.lm_head(hidden.to(dtype=self.lm_head.weight.dtype)).float()
        tokens = logits.argmax(dim=-1)
        return WrappedStepOutput(
            base=base,
            hidden=hidden,
            logits=logits,
            tokens=tokens,
            cllm_backbone_nfe=1,
            corrector_nfe=corrector_nfe,
            initializer_nfe=initializer_nfe,
            corrector_latency_seconds=corrector_latency,
        )

    @torch.inference_mode()
    def rollout(
        self,
        prompt_ids: torch.Tensor,
        initial_tokens: torch.Tensor,
        times: torch.Tensor,
        *,
        max_rounds: int | None = None,
        disable_adapter: bool = False,
    ) -> WrappedRolloutOutput:
        prefill = self.prefill(prompt_ids)
        current = initial_tokens
        flat_times = torch.as_tensor(times, device=current.device)
        if flat_times.ndim == 0:
            if max_rounds is None:
                raise ValueError("max_rounds is required when rollout time is scalar")
            flat_times = flat_times.expand(max_rounds)
        rounds_limit = min(
            int(max_rounds) if max_rounds is not None else len(flat_times), len(flat_times)
        )
        cllm_nfe = corrector_nfe = initializer_nfe = 0
        corrector_latency = 0.0
        converged = False
        rounds = 0
        for round_index in range(rounds_limit):
            output = self(
                prefill,
                current,
                flat_times[round_index],
                round_index=round_index,
                disable_adapter=disable_adapter,
            )
            rounds += 1
            cllm_nfe += output.cllm_backbone_nfe
            corrector_nfe += output.corrector_nfe
            initializer_nfe += output.initializer_nfe
            corrector_latency += output.corrector_latency_seconds
            if torch.equal(output.tokens, current):
                current = output.tokens
                converged = True
                break
            current = output.tokens
        return WrappedRolloutOutput(
            tokens=current,
            rounds=rounds,
            cllm_backbone_nfe=cllm_nfe,
            corrector_nfe=corrector_nfe,
            initializer_nfe=initializer_nfe,
            prompt_prefill_nfe=prefill.prompt_nfe,
            corrector_latency_seconds=corrector_latency,
            converged=converged,
        )
