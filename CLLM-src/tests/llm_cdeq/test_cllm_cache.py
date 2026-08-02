import json

import pytest

from llm_cdeq.cllm_cache import read_official_manifest
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
