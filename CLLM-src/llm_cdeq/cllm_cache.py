from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Mapping

import torch
from safetensors.torch import load_file, save_file


OFFICIAL_CLLM_CACHE_SCHEMA = "llm_cdeq_official_cllm_hidden_cache_v1"


def write_official_shard(path: str | Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in tensors.items()
        },
        str(path),
    )


def read_official_manifest(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != OFFICIAL_CLLM_CACHE_SCHEMA:
        raise ValueError(
            "wrapped trainer requires official CLLM cache schema "
            f"{OFFICIAL_CLLM_CACHE_SCHEMA!r}; got "
            f"{manifest.get('schema_version')!r}. Legacy Abel caches are not valid."
        )
    if manifest.get("operator") != "official_cllm":
        raise ValueError("wrapped trainer refuses a cache not produced by official CLLM")
    return manifest


class OfficialCLLMTrajectoryDataset:
    def __init__(self, manifest_path: str | Path, max_open_shards: int = 2):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = read_official_manifest(self.manifest_path)
        self.shards = list(self.manifest["shards"])
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.total = total
        self.max_open_shards = int(max_open_shards)
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.total

    def _load(self, shard_index: int) -> dict[str, torch.Tensor]:
        if shard_index in self._cache:
            value = self._cache.pop(shard_index)
            self._cache[shard_index] = value
            return value
        value = load_file(
            str(self.root / self.shards[shard_index]["file"]), device="cpu"
        )
        self._cache[shard_index] = value
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[shard_index - 1] if shard_index else 0
        offset = index - previous
        return {key: value[offset] for key, value in self._load(shard_index).items()}

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shard_indices = torch.arange(len(self.shards))
        if shuffle:
            shard_indices = shard_indices[
                torch.randperm(len(shard_indices), generator=generator)
            ]
        for shard_index in shard_indices.tolist():
            shard = self._load(shard_index)
            count = next(iter(shard.values())).shape[0]
            indices = torch.arange(count)
            if shuffle:
                indices = indices[torch.randperm(count, generator=generator)]
            for start in range(0, count, batch_size):
                selected = indices[start : start + batch_size]
                yield {key: value[selected] for key, value in shard.items()}
