import hashlib
import json

import pytest

from llm_cdeq.verify_artifacts import verify_local_dir


def test_hub_local_dir_revision_and_lfs_digest(tmp_path):
    payload = b"weights"
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(payload)
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "CACHEDIR.TAG").write_text(
        "cache metadata, not a Hub artifact\n", encoding="utf-8"
    )
    (metadata / "model.bin.metadata").write_text(
        "revision\n" + hashlib.sha256(payload).hexdigest() + "\n0\n",
        encoding="utf-8",
    )
    report = verify_local_dir(tmp_path, "revision")
    assert report["verified"]
    assert report["total_bytes"] == len(payload)


def test_pinned_manifest_rejects_missing_files(tmp_path):
    payload = b"complete"
    (tmp_path / "present.bin").write_bytes(payload)
    manifest = tmp_path.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "llm_cdeq_artifact_manifest_v1",
                "revision": "pinned",
                "files": [
                    {
                        "path": "present.bin",
                        "size": len(payload),
                        "etag": hashlib.sha256(payload).hexdigest(),
                    },
                    {
                        "path": "missing.bin",
                        "size": 1,
                        "etag": hashlib.sha256(b"x").hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing.bin"):
        verify_local_dir(tmp_path, "pinned", manifest)


def test_pinned_manifest_allows_manual_download_without_hf_metadata(tmp_path):
    payload = b"manual aria2 download"
    (tmp_path / "weight.bin").write_bytes(payload)
    manifest = tmp_path.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "llm_cdeq_artifact_manifest_v1",
                "revision": "pinned",
                "files": [
                    {
                        "path": "weight.bin",
                        "size": len(payload),
                        "etag": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert verify_local_dir(tmp_path, "pinned", manifest)["verified"]
