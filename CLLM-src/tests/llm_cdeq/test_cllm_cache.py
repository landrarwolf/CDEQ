import json
from copy import deepcopy
from pathlib import Path

import pytest

from llm_cdeq.cllm_cache import read_official_manifest
from llm_cdeq.config import (
    CACHE_TIME_GRID_CONTRACT,
    cache_config_digest,
    load_config,
)
from llm_cdeq.runtime import split_for_data_id


def test_wrapped_cache_rejects_legacy_abel_schema(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": "llm_cdeq_hidden_cache_v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Legacy Abel caches"):
        read_official_manifest(path)


def test_official_cache_requires_official_operator(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "llm_cdeq_official_cllm_hidden_cache_v1",
                "operator": "abel_jacobi",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official CLLM"):
        read_official_manifest(path)


def test_pilot_split_fills_exact_disjoint_deterministic_limits():
    def select():
        selected = {"train": set(), "validation": set()}
        limits = {"train": 512, "validation": 128}
        fraction = limits["validation"] / sum(limits.values())
        for index in range(10_000):
            data_id = f"gsm8k-{index}"
            split = split_for_data_id(data_id, 42, fraction)
            if len(selected[split]) < limits[split]:
                selected[split].add(data_id)
            if all(len(selected[name]) == limit for name, limit in limits.items()):
                break
        return selected

    first = select()
    second = select()
    assert first == second
    assert len(first["train"]) == 512
    assert len(first["validation"]) == 128
    assert first["train"].isdisjoint(first["validation"])


def test_cache_digest_only_tracks_effective_cache_recipe():
    config = load_config(
        Path(__file__).parents[2]
        / "configs/llm_cdeq/gsm8k_wrapped_cllm_pilot.yaml"
    )
    digest = cache_config_digest(config)
    tuning_change = deepcopy(config)
    tuning_change["paths"]["cache_dir"] = "cache-b"
    tuning_change["paths"]["output_dir"] = "run-b"
    tuning_change["model"]["corrector_rank"] = 1024
    tuning_change["model"]["corrector_layers"] = 2
    tuning_change["training"]["epochs"] = 10
    tuning_change["training"]["token_ce_weight"] = 0.2
    tuning_change["evaluation"]["endpoint_error_improvement_gate"] = 0.2
    assert cache_config_digest(tuning_change) == digest

    for override in (
        {"train_limit": 64},
        {"validation_limit": 64},
        {"shard_size": 32},
        {"attention_backend": "flash_attention_2"},
    ):
        assert cache_config_digest(config, **override) != digest
    cache_change = deepcopy(config)
    cache_change["upstream"]["cllm_model_revision"] = "model-v2"
    assert cache_config_digest(cache_change) != digest
    cache_change = deepcopy(config)
    cache_change["time"]["terminal"] = 6.0
    assert cache_config_digest(cache_change) != digest
    assert CACHE_TIME_GRID_CONTRACT == "exact_epsilon_to_terminal_v1"
