import json

import pytest

from llm_cdeq.cllm_cache import read_official_manifest


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
