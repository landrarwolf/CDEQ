from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_SCHEMA = "llm_cdeq_config_v1"
CACHE_TIME_GRID_CONTRACT = "exact_epsilon_to_terminal_v1"


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
    rank_key = "corrector_rank" if model.get("operator") == "official_cllm" else "bottleneck_rank"
    if int(model["hidden_size"]) <= 0 or int(model[rank_key]) <= 0:
        raise ValueError(f"hidden_size and {rank_key} must be positive")
    if int(model["block_size"]) <= 0:
        raise ValueError("block_size must be positive")
    if model.get("operator") == "official_cllm":
        if model.get("cache_schema") != "llm_cdeq_official_cllm_hidden_cache_v1":
            raise ValueError("official CLLM config must use the official hidden cache schema")
        for key in ("corrector_layers", "corrector_heads", "corrector_ffn_size"):
            if int(model[key]) <= 0:
                raise ValueError(f"model.{key} must be positive")
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
    if training.get("checkpoint_selection", "online_endpoint_error") not in (
        "online_endpoint_error",
        "token_exact_any",
    ):
        raise ValueError("unsupported training.checkpoint_selection")


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_config_digest(
    config: Mapping[str, Any],
    *,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    shard_size: int | None = None,
    attention_backend: str | None = None,
) -> str:
    """Hash only the effective inputs that determine an official CLLM cache."""
    upstream = config["upstream"]
    model = config["model"]
    time = config["time"]
    training = config["training"]
    recipe = {
        "upstream": {
            key: value
            for key, value in upstream.items()
            if key == "code_revision" or key.endswith(("_id", "_revision"))
        },
        "model": {
            "operator": model.get("operator"),
            "hidden_size": int(model["hidden_size"]),
            "block_size": int(model["block_size"]),
            "max_trajectory_states": int(model["max_trajectory_states"]),
            "cache_schema": model.get("cache_schema"),
        },
        "time": {
            "epsilon": float(time["epsilon"]),
            "terminal": float(time["terminal"]),
            "rho": float(time["rho"]),
            "grid_contract": CACHE_TIME_GRID_CONTRACT,
        },
        "data": {
            "seed": int(training["seed"]),
            "train_limit": int(
                training["train_limit"] if train_limit is None else train_limit
            ),
            "validation_limit": int(
                training.get("validation_limit", 0)
                if validation_limit is None
                else validation_limit
            ),
            "shard_size": int(
                training["shard_size"] if shard_size is None else shard_size
            ),
        },
        "attention_backend": str(
            config["evaluation"]["attention_backend"]
            if attention_backend is None
            else attention_backend
        ),
    }
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_repo_path(config: Mapping[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = Path(str(config.get("_config_path", Path.cwd())))
    return (config_path.parent.parent.parent / path).resolve()
