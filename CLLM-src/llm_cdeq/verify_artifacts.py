from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARTIFACT_MANIFEST_SCHEMA = "llm_cdeq_artifact_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an hf local-dir against revision metadata")
    parser.add_argument("--root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--manifest",
        help="pinned file/size/etag manifest; also rejects missing or extra files",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def digest_file(path: Path, expected: str) -> str:
    if len(expected) == 64:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if len(expected) == 40:
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    raise ValueError(f"unsupported Hub etag length for {path}: {expected!r}")


def read_expected_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported artifact manifest schema: {manifest.get('schema_version')!r}"
        )
    return manifest


def verify_local_dir(
    root: str | Path,
    revision: str,
    expected_manifest: str | Path | None = None,
) -> dict:
    root = Path(root)
    cache_root = root / ".cache"
    metadata_root = root / ".cache" / "huggingface" / "download"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and cache_root not in path.parents
    )
    expected_by_path: dict[str, dict] | None = None
    missing: list[str] = []
    extra: list[str] = []
    if expected_manifest is not None:
        expected = read_expected_manifest(expected_manifest)
        if expected["revision"] != revision:
            raise ValueError(
                f"manifest revision mismatch: {expected['revision']} != {revision}"
            )
        expected_by_path = {record["path"]: record for record in expected["files"]}
        actual_paths = {str(path.relative_to(root)) for path in files}
        expected_paths = set(expected_by_path)
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)

    records: list[dict] = []
    for path in files:
        relative = path.relative_to(root)
        pinned = expected_by_path.get(str(relative)) if expected_by_path else None
        if expected_by_path is not None and pinned is None:
            continue
        metadata = metadata_root / relative.parent / f"{relative.name}.metadata"
        if pinned is None and not metadata.exists():
            raise FileNotFoundError(f"missing Hugging Face metadata for {relative}")
        etag = pinned["etag"] if pinned else None
        if metadata.exists():
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if len(lines) < 2:
                raise ValueError(f"invalid Hugging Face metadata for {relative}")
            recorded_revision, recorded_etag = lines[:2]
            if recorded_revision != revision:
                raise ValueError(
                    f"revision mismatch for {relative}: {recorded_revision} != {revision}"
                )
            if etag is not None and recorded_etag != etag:
                raise ValueError(
                    f"etag mismatch for {relative}: {recorded_etag} != {etag}"
                )
            etag = recorded_etag
        if pinned is not None and path.stat().st_size != int(pinned["size"]):
            records.append(
                {
                    "file": str(relative),
                    "size": path.stat().st_size,
                    "expected_size": int(pinned["size"]),
                    "etag": etag,
                    "digest": None,
                    "verified": False,
                }
            )
            continue
        actual = digest_file(path, etag)
        records.append(
            {
                "file": str(relative),
                "size": path.stat().st_size,
                "etag": etag,
                "digest": actual,
                "verified": actual == etag,
            }
        )
    failed = [record["file"] for record in records if not record["verified"]]
    report = {
        "root": str(root.resolve()),
        "revision": revision,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "manifest": str(Path(expected_manifest).resolve()) if expected_manifest else None,
        "verified": not failed and not missing and not extra,
        "failed": failed,
        "missing": missing,
        "extra": extra,
    }
    if failed or missing or extra:
        raise ValueError(
            f"artifact verification failed: failed={failed}, missing={missing}, extra={extra}"
        )
    return report


def main() -> None:
    args = parse_args()
    report = verify_local_dir(args.root, args.revision, args.manifest)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
