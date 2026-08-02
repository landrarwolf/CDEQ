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
from .runtime import eos_mask, iter_json_array, split_for_data_id, stable_hash, unbatch_ids
from .time import rho_time_grid


class OfficialCLLMAlignmentError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical hidden trajectories from the official CLLM operator"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
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


class OfficialSplitWriter:
    def __init__(self, root: Path, split: str, shard_size: int):
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.records: list[dict[str, torch.Tensor]] = []
        self.metadata: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.count = 0
        self.metadata_path = self.root / "metadata.jsonl"
        self.metadata_handle = self.metadata_path.open("w", encoding="utf-8")

    def add(self, record: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
        self.records.append(record)
        self.metadata.append(metadata)
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        shard_name = f"shard-{len(self.shards):05d}.safetensors"
        shard_path = self.root / shard_name
        tensors = {
            key: torch.stack([record[key] for record in self.records])
            for key in self.records[0]
        }
        write_official_shard(shard_path, tensors)
        for item in self.metadata:
            self.metadata_handle.write(json.dumps(item, separators=(",", ":")) + "\n")
        count = len(self.records)
        self.shards.append(
            {"file": shard_name, "count": count, "sha256": _sha256(shard_path)}
        )
        self.count += count
        self.records.clear()
        self.metadata.clear()

    def close(self, common: dict[str, Any]) -> Path:
        self.flush()
        self.metadata_handle.close()
        manifest = {
            **common,
            "split": self.split,
            "count": self.count,
            "shards": self.shards,
            "metadata_file": self.metadata_path.name,
            "metadata_sha256": _sha256(self.metadata_path),
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["model"].get("operator") != "official_cllm":
        raise ValueError("prepare_cllm_states requires model.operator=official_cllm")
    expected_id = config["upstream"].get("cllm_model_id")
    if expected_id != OFFICIAL_CLLM_ID:
        raise ValueError(f"official CLLM id must be {OFFICIAL_CLLM_ID!r}")

    training = config["training"]
    if args.limit is not None and (
        args.train_limit is not None or args.validation_limit is not None
    ):
        raise ValueError("--limit cannot be combined with split-specific limits")
    if args.limit is not None:
        train_limit, validation_limit = args.limit, 0
    else:
        train_limit = args.train_limit or int(training["train_limit"])
        validation_limit = (
            args.validation_limit
            if args.validation_limit is not None
            else int(training.get("validation_limit", 0))
        )
    if train_limit <= 0 or validation_limit < 0:
        raise ValueError("train limit must be positive and validation limit non-negative")

    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    if cache_root.exists() and any(cache_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"cache directory is not empty: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
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
    shard_size = int(training["shard_size"])
    split_names = ("train", "validation") if validation_limit else ("train",)
    writers = {
        split: OfficialSplitWriter(cache_root, split, shard_size) for split in split_names
    }
    limits = {"train": train_limit, "validation": validation_limit}
    accepted = {"train": 0, "validation": 0}
    accepted_data_ids = {"train": set(), "validation": set()}
    rejected: dict[str, int] = {}
    validation_fraction = validation_limit / (train_limit + validation_limit)
    progress = tqdm(total=train_limit + validation_limit, desc="official CLLM blocks")
    for source_index, raw in enumerate(iter_json_array(source)):
        if accepted["train"] >= train_limit and accepted["validation"] >= validation_limit:
            break
        try:
            example = normalize_initial_example(raw, operator.block_size)
            split = (
                split_for_data_id(
                    example["data_id"], int(training["seed"]), validation_fraction
                )
                if validation_limit
                else "train"
            )
            if (
                accepted[split] >= limits[split]
                or example["data_id"] in accepted_data_ids[split]
            ):
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
        item["split"] = split
        accepted_data_ids[split].add(example["data_id"])
        accepted[split] += 1
        writers[split].add(record, item)
        progress.update(1)
        progress.set_postfix(train=accepted["train"], validation=accepted["validation"])
    progress.close()
    if accepted != limits:
        raise RuntimeError(
            f"could not fill requested official CLLM splits: {accepted} != {limits}"
        )

    overlap = accepted_data_ids["train"] & accepted_data_ids["validation"]
    if overlap:
        raise RuntimeError("official CLLM train/validation data_id overlap")
    split_hash = stable_hash(
        sorted(
            (data_id, split)
            for split, data_ids in accepted_data_ids.items()
            for data_id in data_ids
        )
    )
    common = {
        "schema_version": OFFICIAL_CLLM_CACHE_SCHEMA,
        "operator": "official_cllm",
        "created_unix": time.time(),
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
        "data_split_hash": split_hash,
        "split_counts": accepted,
        "data_id_overlap": 0,
        "rejected": rejected,
    }
    manifests = {split: str(writer.close(common)) for split, writer in writers.items()}
    summary = {
        "manifests": manifests,
        "accepted": accepted,
        "data_id_overlap": 0,
        "data_split_hash": split_hash,
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
