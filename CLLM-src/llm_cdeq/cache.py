from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Mapping

import torch
from safetensors.torch import load_file, save_file


CACHE_SCHEMA = "llm_cdeq_hidden_cache_v1"


def write_shard(path: str | Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(contiguous, str(path))


def write_manifest(path: str | Path, manifest: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_manifest(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(
            f"Cache {path} has schema {manifest.get('schema_version')!r}; "
            f"expected {CACHE_SCHEMA!r}"
        )
    return manifest


class HiddenTrajectoryDataset:
    """Read-only random access to sharded hidden trajectories."""

    def __init__(self, manifest_path: str | Path, max_open_shards: int = 2):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = read_manifest(self.manifest_path)
        self.shards = self.manifest["shards"]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self.total = total
        self.max_open_shards = max_open_shards
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if shard_index in self._cache:
            value = self._cache.pop(shard_index)
            self._cache[shard_index] = value
            return value
        value = load_file(str(self.root / self.shards[shard_index]["file"]), device="cpu")
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
        shard = self._load_shard(shard_index)
        return {name: value[offset] for name, value in shard.items()}

    def iter_shards(
        self,
        *,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        indices = torch.arange(len(self.shards))
        if shuffle:
            indices = indices[torch.randperm(len(indices), generator=generator)]
        for index in indices.tolist():
            yield self._load_shard(index)


def iter_shard_batches(
    dataset: HiddenTrajectoryDataset,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for shard in dataset.iter_shards(shuffle=shuffle, generator=generator):
        count = next(iter(shard.values())).shape[0]
        indices = torch.arange(count)
        if shuffle:
            indices = indices[torch.randperm(count, generator=generator)]
        for start in range(0, count, batch_size):
            selected = indices[start : start + batch_size]
            yield {name: value[selected] for name, value in shard.items()}

