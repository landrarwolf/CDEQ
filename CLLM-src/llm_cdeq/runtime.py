from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_for_data_id(data_id: Any, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{data_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def iter_json_array(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    try:
        import ijson
    except ImportError:
        with path.open("r", encoding="utf-8") as handle:
            yield from json.load(handle)
        return
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


def unbatch_ids(value: Any) -> list[int]:
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or (value and isinstance(value[0], list)):
        raise ValueError("expected a single token-id sequence")
    return [int(token) for token in value]


def eos_mask(tokens: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    """Keep tokens through the first EOS, masking every later position."""
    if tokens.ndim < 1:
        raise ValueError("tokens must have at least one dimension")
    if eos_token_id is None:
        return torch.ones_like(tokens, dtype=torch.bool)
    is_eos = tokens.eq(eos_token_id)
    eos_seen_before = is_eos.cumsum(dim=-1) - is_eos.to(torch.int64)
    return eos_seen_before.eq(0)


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str,
    *,
    floating_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        dtype = floating_dtype if floating_dtype and torch.is_floating_point(value) else None
        moved[name] = value.to(device=device, dtype=dtype, non_blocking=True)
    return moved


def batched(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

