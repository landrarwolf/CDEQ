from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_SCHEMA = "llm_cdeq_config_v1"


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(
            f"Config {path} has schema {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA!r}"
        )
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ("upstream", "paths", "model", "time", "training", "evaluation"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing config section: {section}")

    model = config["model"]
    time = config["time"]
    training = config["training"]
    if int(model["hidden_size"]) <= 0 or int(model["bottleneck_rank"]) <= 0:
        raise ValueError("hidden_size and bottleneck_rank must be positive")
    if int(model["block_size"]) <= 0:
        raise ValueError("block_size must be positive")
    if not 0 < float(time["epsilon"]) < float(time["terminal"]):
        raise ValueError("time must satisfy 0 < epsilon < terminal")
    if float(time["rho"]) <= 0 or float(time["q"]) <= 1:
        raise ValueError("rho must be positive and q must be > 1")
    if int(time["d"]) <= 0:
        raise ValueError("time.d must be positive")
    if not 0 <= float(training["local_weight"]) <= 1:
        raise ValueError("local_weight must be in [0, 1]")
    if not 0 <= float(training["endpoint_weight"]) <= 1:
        raise ValueError("endpoint_weight must be in [0, 1]")
    if abs(
        float(training["local_weight"]) + float(training["endpoint_weight"]) - 1.0
    ) > 1e-6:
        raise ValueError("local_weight + endpoint_weight must equal 1")


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_repo_path(config: Mapping[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = Path(str(config.get("_config_path", Path.cwd())))
    return (config_path.parent.parent.parent / path).resolve()

