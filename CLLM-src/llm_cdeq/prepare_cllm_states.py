from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from .cllm_cache import OFFICIAL_CLLM_CACHE_SCHEMA, write_official_shard
from .cllm_step import OFFICIAL_CLLM_ID, OfficialCLLMSingleStep, parameter_checksum
from .config import config_digest, load_config, public_config, resolve_repo_path
from .runtime import eos_mask, iter_json_array, stable_hash, unbatch_ids
from .time import rho_time_grid


class OfficialCLLMAlignmentError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical hidden trajectories from the official CLLM operator"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_initial_example(raw: dict[str, Any], block_size: int) -> dict[str, Any]:
    prompt = unbatch_ids(raw["prompt_ids"])
    trajectory = raw.get("answer_trajectory_ids")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("record has no initial Jacobi state")
    initial = unbatch_ids(trajectory[0])
    if len(initial) != block_size:
        raise ValueError("initial token block does not match configured block size")
    data_id = raw.get("data_id")
    if data_id is None:
        raise ValueError("record is missing data_id")
    return {
        "data_id": str(data_id),
        "jacobian_itr_id": raw.get("jacobian_itr_id"),
        "prompt_ids": prompt,
        "initial_tokens": initial,
    }


@torch.inference_mode()
def encode_official_trajectory(
    operator: OfficialCLLMSingleStep,
    example: dict[str, Any],
    *,
    max_states: int,
    epsilon: float,
    terminal: float,
    rho: float,
    eos_token_id: int | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prompt_ids = torch.tensor(
        [example["prompt_ids"]], dtype=torch.long, device=device
    )
    current = torch.tensor(
        [example["initial_tokens"]], dtype=torch.long, device=device
    )
    prefill = operator.prefill(prompt_ids)
    if not torch.equal(current[:, :1], prefill.first_token):
        raise OfficialCLLMAlignmentError(
            "initial y0 does not contain the official prompt-prefill first token"
        )

    hidden_states: list[torch.Tensor] = []
    input_tokens: list[torch.Tensor] = []
    output_tokens: list[torch.Tensor] = []
    alignment_tokens = 0
    alignment_total = 0
    converged = False
    for _ in range(max_states):
        step = operator(prefill, current)
        decoded = operator.lm_head(step.canonical_hidden).float().argmax(dim=-1)
        aligned = decoded.eq(step.tokens)
        alignment_tokens += int(aligned.sum())
        alignment_total += aligned.numel()
        if not bool(aligned.all()):
            raise OfficialCLLMAlignmentError(
                "LMHead(canonical_hidden) disagrees with official CLLM single step"
            )
        hidden_states.append(step.canonical_hidden[0].to(torch.bfloat16).cpu())
        input_tokens.append(current[0].cpu())
        output_tokens.append(step.tokens[0].cpu())
        if torch.equal(step.tokens, current):
            converged = True
            break
        current = step.tokens
    if not converged:
        raise OfficialCLLMAlignmentError(
            f"official CLLM trajectory did not reach a fixed point in {max_states} states"
        )

    count = len(hidden_states)
    block_size = operator.block_size
    hidden_size = operator.hidden_size
    canonical_hidden = torch.zeros(
        max_states, block_size, hidden_size, dtype=torch.bfloat16
    )
    padded_inputs = torch.zeros(max_states, block_size, dtype=torch.long)
    padded_outputs = torch.zeros(max_states, block_size, dtype=torch.long)
    state_mask = torch.zeros(max_states, dtype=torch.bool)
    canonical_hidden[:count] = torch.stack(hidden_states)
    padded_inputs[:count] = torch.stack(input_tokens)
    padded_outputs[:count] = torch.stack(output_tokens)
    state_mask[:count] = True
    endpoint_hidden = hidden_states[-1]
    endpoint_tokens = output_tokens[-1]
    token_mask = eos_mask(endpoint_tokens, eos_token_id)
    state_token_mask = torch.zeros(max_states, block_size, dtype=torch.bool)
    for index, tokens in enumerate(output_tokens):
        state_token_mask[index] = eos_mask(tokens, eos_token_id)
    time_grid = torch.full((max_states,), float("nan"), dtype=torch.float32)
    time_grid[:count] = rho_time_grid(
        count, epsilon=epsilon, terminal=terminal, rho=rho
    )
    record = {
        "canonical_hidden": canonical_hidden,
        "input_tokens": padded_inputs,
        "output_tokens": padded_outputs,
        "state_mask": state_mask,
        "state_token_mask": state_token_mask,
        "endpoint_hidden": endpoint_hidden,
        "endpoint_tokens": endpoint_tokens,
        "eos_mask": token_mask,
        "time_grid": time_grid,
    }
    metadata = {
        "data_id": example["data_id"],
        "jacobian_itr_id": example["jacobian_itr_id"],
        "prompt_ids": example["prompt_ids"],
        "initial_tokens_sha256": stable_hash(example["initial_tokens"]),
        "trajectory_length": count,
        "alignment_tokens": alignment_tokens,
        "alignment_total": alignment_total,
        "alignment_rate": alignment_tokens / max(alignment_total, 1),
        "fixed_point": True,
        "prompt_cache_lengths": list(prefill.prompt_cache_lengths),
    }
    return record, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    config = load_config(args.config)
    if config["model"].get("operator") != "official_cllm":
        raise ValueError("prepare_cllm_states requires model.operator=official_cllm")
    expected_id = config["upstream"].get("cllm_model_id")
    if expected_id != OFFICIAL_CLLM_ID:
        raise ValueError(f"official CLLM id must be {OFFICIAL_CLLM_ID!r}")

    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    if cache_root.exists() and any(cache_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"cache directory is not empty: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    split_root = cache_root / "train"
    split_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    attention_backend = args.attention_backend or config["evaluation"]["attention_backend"]
    model_path = resolve_repo_path(config, config["paths"]["cllm_model"])
    operator = OfficialCLLMSingleStep.from_pretrained(
        model_path,
        block_size=int(config["model"]["block_size"]),
        attention_backend=attention_backend,
        device=device,
    )
    checksum_before = parameter_checksum(operator.model)
    save_file(
        {"weight": operator.lm_head.weight.detach().to(torch.bfloat16).cpu().contiguous()},
        str(cache_root / "lm_head.safetensors"),
    )

    max_states = int(config["model"]["max_trajectory_states"])
    source = resolve_repo_path(config, config["paths"]["trajectory_json"])
    shard_size = int(config["training"]["shard_size"])
    accepted: list[dict[str, torch.Tensor]] = []
    metadata: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    seen_data_ids: set[str] = set()
    rejected: dict[str, int] = {}
    metadata_path = split_root / "metadata.jsonl"
    metadata_handle = metadata_path.open("w", encoding="utf-8")

    def flush() -> None:
        if not accepted:
            return
        shard_name = f"shard-{len(shards):05d}.safetensors"
        shard_path = split_root / shard_name
        tensors = {
            key: torch.stack([record[key] for record in accepted])
            for key in accepted[0]
        }
        write_official_shard(shard_path, tensors)
        for item in metadata:
            metadata_handle.write(json.dumps(item, separators=(",", ":")) + "\n")
        shards.append(
            {"file": shard_name, "count": len(accepted), "sha256": _sha256(shard_path)}
        )
        accepted.clear()
        metadata.clear()

    progress = tqdm(total=args.limit, desc="official CLLM blocks")
    for source_index, raw in enumerate(iter_json_array(source)):
        if len(seen_data_ids) >= args.limit:
            break
        try:
            example = normalize_initial_example(raw, operator.block_size)
            if example["data_id"] in seen_data_ids:
                continue
            record, item = encode_official_trajectory(
                operator,
                example,
                max_states=max_states,
                epsilon=float(config["time"]["epsilon"]),
                terminal=float(config["time"]["terminal"]),
                rho=float(config["time"]["rho"]),
                eos_token_id=operator.model.config.eos_token_id,
                device=device,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            key = type(error).__name__
            rejected[key] = rejected.get(key, 0) + 1
            continue
        item["source_index"] = source_index
        seen_data_ids.add(example["data_id"])
        accepted.append(record)
        metadata.append(item)
        progress.update(1)
        if len(accepted) >= shard_size:
            flush()
    progress.close()
    flush()
    metadata_handle.close()
    if len(seen_data_ids) != args.limit:
        raise RuntimeError(
            f"built only {len(seen_data_ids)} of {args.limit} requested official CLLM blocks"
        )

    manifest = {
        "schema_version": OFFICIAL_CLLM_CACHE_SCHEMA,
        "operator": "official_cllm",
        "created_unix": time.time(),
        "split": "train",
        "count": len(seen_data_ids),
        "shards": shards,
        "metadata_file": metadata_path.name,
        "metadata_sha256": _sha256(metadata_path),
        "lm_head_file": "../lm_head.safetensors",
        "lm_head_sha256": _sha256(cache_root / "lm_head.safetensors"),
        "shape": {
            "canonical_hidden": [
                max_states,
                operator.block_size,
                operator.hidden_size,
            ]
        },
        "dtype": "bfloat16",
        "cache_hidden_token_alignment": 1.0,
        "attention_backend": attention_backend,
        "cllm_model_id": expected_id,
        "cllm_model_revision": config["upstream"]["cllm_model_revision"],
        "tokenizer_id": config["upstream"].get("tokenizer_id"),
        "tokenizer_revision": config["upstream"].get("tokenizer_revision"),
        "backbone_parameter_count": operator.backbone_parameter_count,
        "backbone_checksum": checksum_before,
        "eos_token_id": operator.model.config.eos_token_id,
        "config": public_config(config),
        "config_digest": config_digest(config),
        "data_split_hash": stable_hash(sorted(seen_data_ids)),
        "rejected": rejected,
    }
    manifest_path = split_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "manifest": str(manifest_path),
        "accepted": len(seen_data_ids),
        "rejected": rejected,
        "alignment": 1.0,
        "backbone_checksum": checksum_before,
    }
    (cache_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
