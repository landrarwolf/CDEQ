import json

import torch

from llm_cdeq.cache import CACHE_SCHEMA, HiddenTrajectoryDataset, write_manifest, write_shard
from llm_cdeq.runtime import eos_mask, split_for_data_id


def test_eos_mask_keeps_first_eos_and_masks_tail():
    tokens = torch.tensor([[1, 2, 9, 4, 9], [1, 2, 3, 4, 5]])
    expected = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    assert torch.equal(eos_mask(tokens, 9), expected)


def test_split_is_stable_per_data_id():
    assert split_for_data_id("same", 42, 0.1) == split_for_data_id("same", 42, 0.1)


def test_shard_manifest_round_trip(tmp_path):
    write_shard(
        tmp_path / "shard.safetensors",
        {"states": torch.arange(24).reshape(3, 2, 4), "mask": torch.ones(3, 2, dtype=torch.bool)},
    )
    write_manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": CACHE_SCHEMA,
            "shards": [{"file": "shard.safetensors", "count": 3}],
        },
    )
    dataset = HiddenTrajectoryDataset(tmp_path / "manifest.json")
    assert len(dataset) == 3
    torch.testing.assert_close(dataset[2]["states"], torch.arange(16, 24).reshape(2, 4))

