from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


OFFICIAL_CLLM_ID = "cllm/consistency-llm-7b-math"


def _as_legacy_cache(cache: Any) -> tuple:
    if cache is None:
        return ()
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple(tuple(value.detach() for value in layer) for layer in cache)


def cache_sequence_lengths(cache: tuple) -> tuple[int, ...]:
    """Return the cached sequence length for every decoder layer."""
    return tuple(int(layer[0].shape[-2]) for layer in cache)


def parameter_checksum(module: nn.Module) -> str:
    """Hash every named parameter without assembling a second model-sized buffer."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CLLMPrefill:
    prompt_ids: torch.Tensor
    prompt_last_hidden: torch.Tensor
    first_token: torch.Tensor
    prompt_cache: tuple
    prompt_cache_lengths: tuple[int, ...]
    prompt_nfe: int = 1


@dataclass(frozen=True)
class CLLMStepOutput:
    canonical_hidden: torch.Tensor
    logits: torch.Tensor
    tokens: torch.Tensor
    block_hidden: torch.Tensor
    prompt_cache_lengths_before: tuple[int, ...]
    prompt_cache_lengths_after: tuple[int, ...]
    backbone_nfe: int = 1


class OfficialCLLMSingleStep(nn.Module):
    """One canonical Jacobi iteration of the frozen official CLLM checkpoint.

    The reusable prompt cache is stored as an immutable legacy tuple.  Every
    block step creates a transient generation cache inside Transformers; that
    transient cache is discarded with the return value, so candidate block KV
    entries can never leak into a later Jacobi iteration.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        block_size: int = 16,
        model_source: str | Path | None = None,
        enforce_official: bool = True,
    ):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise TypeError("official CLLM must expose .model and .lm_head")
        if enforce_official:
            source = str(model_source or getattr(model.config, "_name_or_path", ""))
            normalized = source.rstrip("/")
            if normalized != OFFICIAL_CLLM_ID and Path(normalized).name != "consistency-llm-7b-math":
                raise ValueError(
                    "wrapped CDEQ+ requires cllm/consistency-llm-7b-math; "
                    f"refusing model source {source!r}"
                )
        self.model = model
        self.block_size = int(block_size)
        self.model_source = str(model_source or getattr(model.config, "_name_or_path", ""))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def backbone_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        block_size: int = 16,
        torch_dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "sdpa",
        device: str | torch.device | None = None,
    ) -> "OfficialCLLMSingleStep":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attention_backend,
        )
        if device is not None:
            model = model.to(device)
        return cls(
            model,
            block_size=block_size,
            model_source=model_path,
            enforce_official=True,
        )

    @torch.inference_mode()
    def prefill(
        self,
        prompt_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> CLLMPrefill:
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1:
            raise ValueError("prompt_ids must have shape [batch, prompt_length >= 1]")
        output = self.model.model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        prompt_cache = _as_legacy_cache(output.past_key_values)
        prompt_last_hidden = output.last_hidden_state[:, -1:].detach()
        first_token = self.lm_head(prompt_last_hidden).float().argmax(dim=-1)
        return CLLMPrefill(
            prompt_ids=prompt_ids.detach(),
            prompt_last_hidden=prompt_last_hidden,
            first_token=first_token,
            prompt_cache=prompt_cache,
            prompt_cache_lengths=cache_sequence_lengths(prompt_cache),
        )

    @torch.inference_mode()
    def forward(self, prefill: CLLMPrefill, current_tokens: torch.Tensor) -> CLLMStepOutput:
        if current_tokens.ndim != 2 or current_tokens.shape[1] != self.block_size:
            raise ValueError(
                f"current_tokens must have shape [batch, {self.block_size}]"
            )
        if current_tokens.shape[0] != prefill.prompt_ids.shape[0]:
            raise ValueError("prompt and current token batches must match")
        if not torch.equal(current_tokens[:, :1], prefill.first_token):
            raise ValueError(
                "official Jacobi state invariant violated: block token 0 must "
                "equal the prompt-prefill first token"
            )
        lengths_before = cache_sequence_lengths(prefill.prompt_cache)
        if lengths_before != prefill.prompt_cache_lengths:
            raise RuntimeError("stored prompt cache changed before the CLLM step")

        output = self.model.model(
            input_ids=current_tokens,
            past_key_values=prefill.prompt_cache,
            use_cache=True,
            return_dict=True,
        )
        block_hidden = output.last_hidden_state
        canonical_hidden = torch.cat(
            (prefill.prompt_last_hidden, block_hidden[:, :-1]), dim=1
        )
        logits = self.lm_head(canonical_hidden).float()
        tokens = logits.argmax(dim=-1)

        # This is the exact shift in the official jacobi_forward generation
        # loop.  Equality proves that canonical_hidden is aligned with the
        # official single-step tokens.
        official_shift = torch.cat(
            (
                current_tokens[:, :1],
                self.lm_head(block_hidden[:, :-1]).float().argmax(dim=-1),
            ),
            dim=1,
        )
        if not torch.equal(tokens, official_shift):
            raise RuntimeError("canonical hidden is not aligned with official Jacobi shift")

        lengths_after = cache_sequence_lengths(prefill.prompt_cache)
        if lengths_after != lengths_before:
            raise RuntimeError("candidate block KV polluted the reusable prompt cache")
        return CLLMStepOutput(
            canonical_hidden=canonical_hidden,
            logits=logits,
            tokens=tokens,
            block_hidden=block_hidden,
            prompt_cache_lengths_before=lengths_before,
            prompt_cache_lengths_after=lengths_after,
        )
