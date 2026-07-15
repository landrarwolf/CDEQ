from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from .cache import CACHE_SCHEMA, write_manifest, write_shard
from .config import config_digest, load_config, public_config, resolve_repo_path
from .runtime import eos_mask, iter_json_array, stable_hash, unbatch_ids
from .time import rho_time_grid


class HiddenTokenAlignmentError(ValueError):
    pass


def plan_grouped_split(
    group_counts: dict[str, int],
    *,
    seed: int,
    validation_fraction: float,
    validation_minimum: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign complete data-id groups using a deterministic group-level split.

    Augmented Jacobi datasets can contain a few enormous data-id groups.  A
    record-count knapsack can therefore leave the training side with only one
    or two questions even when thousands of groups exist.  Splitting the
    hashed group order itself keeps question diversity while preserving the
    no-leakage guarantee.
    """
    if not group_counts or not 0 < validation_fraction < 1:
        raise ValueError("group counts must be non-empty and validation fraction in (0, 1)")
    if validation_minimum < 1:
        raise ValueError("validation minimum must be positive")
    total = sum(int(count) for count in group_counts.values())
    if len(group_counts) < 2:
        raise ValueError("grouped split requires at least two data-id groups")
    ordered = sorted(
        group_counts,
        key=lambda data_id: hashlib.sha256(f"{seed}:{data_id}".encode("utf-8")).digest(),
    )
    group_target = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * validation_fraction)),
    )
    validation_ids = set(ordered[:group_target])
    validation_records = sum(int(group_counts[data_id]) for data_id in validation_ids)
    for data_id in ordered[group_target:]:
        if validation_records >= validation_minimum or len(validation_ids) >= len(ordered) - 1:
            break
        validation_ids.add(data_id)
        validation_records += int(group_counts[data_id])
    if validation_records < validation_minimum:
        raise ValueError("validation groups contain too few records")
    assignment = {
        data_id: "validation" if data_id in validation_ids else "train"
        for data_id in group_counts
    }
    record_target = max(validation_minimum, round(total * validation_fraction))
    return assignment, {
        "total_records": total,
        "train_records": total - validation_records,
        "validation_records": validation_records,
        "train_groups": len(group_counts) - len(validation_ids),
        "validation_groups": len(validation_ids),
        "validation_group_target": group_target,
        "validation_record_target": record_target,
        "validation_overshoot": validation_records - record_target,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned Abel hidden-state caches")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-alignment", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_example(raw: dict[str, Any], block_size: int) -> dict[str, Any]:
    prompt = unbatch_ids(raw["prompt_ids"])
    trajectory = [unbatch_ids(value) for value in raw["answer_trajectory_ids"]]
    if len(trajectory) < 2:
        raise ValueError("trajectory has fewer than two states")
    if any(len(state) != block_size for state in trajectory):
        raise ValueError("trajectory block length does not match config")
    return {
        "data_id": raw.get("data_id"),
        "jacobian_itr_id": raw.get("jacobian_itr_id"),
        "prompt": prompt,
        "trajectory": trajectory,
    }


def shifted_hidden_slice(
    hidden: torch.Tensor, prompt_length: int, block_size: int
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, sequence, hidden]")
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    shifted = hidden[:, prompt_length - 1 : prompt_length + block_size - 1]
    if shifted.shape[1] != block_size:
        raise ValueError("hidden sequence is too short for shifted block slice")
    return shifted


def canonical_token_key(tokens: torch.Tensor, eos_token_id: int | None) -> tuple[int, ...]:
    values = [int(token) for token in tokens.tolist()]
    if eos_token_id is not None and eos_token_id in values:
        values = values[: values.index(eos_token_id) + 1]
    return tuple(values)


def deduplicate_token_states(
    trajectory: list[list[int]], eos_token_id: int | None
) -> torch.Tensor:
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for state in trajectory:
        tensor = torch.tensor(state, dtype=torch.long)
        key = canonical_token_key(tensor, eos_token_id)
        if key not in seen:
            seen.add(key)
            unique.append(state)
    return torch.tensor(unique, dtype=torch.long)


def recover_aligned_chain(
    candidate_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    endpoint_tokens: torch.Tensor,
    *,
    eos_token_id: int | None,
    max_states: int,
) -> list[int]:
    """Recover the longest frozen-Jacobi path from interleaved augmented states."""
    keys = [canonical_token_key(tokens, eos_token_id) for tokens in candidate_tokens]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate token states must already be deduplicated")
    index_by_key = {key: index for index, key in enumerate(keys)}
    endpoint_key = canonical_token_key(endpoint_tokens, eos_token_id)
    if endpoint_key not in index_by_key:
        raise HiddenTokenAlignmentError("endpoint is missing from candidate states")
    endpoint_index = index_by_key[endpoint_key]
    successor = {
        index: index_by_key.get(canonical_token_key(prediction, eos_token_id))
        for index, prediction in enumerate(predicted_tokens)
    }
    if successor[endpoint_index] != endpoint_index:
        raise HiddenTokenAlignmentError(
            "endpoint hidden state is not an identity token boundary"
        )

    best: list[int] = []
    for start in range(len(keys)):
        path: list[int] = []
        seen: set[int] = set()
        current: int | None = start
        while current is not None and current not in seen:
            path.append(current)
            seen.add(current)
            if current == endpoint_index:
                if len(path) > len(best):
                    best = path
                break
            current = successor[current]
    if len(best) < 2:
        raise HiddenTokenAlignmentError("no aligned Jacobi chain reaches the endpoint")
    if len(best) > max_states:
        raise ValueError(
            f"aligned chain has {len(best)} states; cache supports {max_states}"
        )
    return best


@torch.inference_mode()
def encode_trajectory(
    model,
    example: dict[str, Any],
    *,
    block_size: int,
    state_batch_size: int,
    max_states: int,
    epsilon: float,
    terminal: float,
    rho: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prompt = example["prompt"]
    raw_trajectory = example["trajectory"]
    prompt_length = len(prompt)
    candidate_tokens = deduplicate_token_states(
        raw_trajectory, model.config.eos_token_id
    )
    hidden_states: list[torch.Tensor] = []
    predicted_tokens: list[torch.Tensor] = []

    for start in range(0, len(candidate_tokens), state_batch_size):
        state_batch = candidate_tokens[start : start + state_batch_size].tolist()
        input_ids = torch.tensor(
            [prompt + state for state in state_batch], dtype=torch.long, device=device
        )
        output = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        shifted = shifted_hidden_slice(output.last_hidden_state, prompt_length, block_size)
        if shifted.shape[1:] != (block_size, model.config.hidden_size):
            raise ValueError(f"unexpected hidden slice shape {tuple(shifted.shape)}")
        hidden_states.extend(shifted.to(torch.bfloat16).cpu().unbind(0))
        logits = model.lm_head(shifted).float()
        predicted_tokens.extend(logits.argmax(dim=-1).cpu().unbind(0))

    predictions = torch.stack(predicted_tokens)
    endpoint_raw = torch.tensor(raw_trajectory[-1], dtype=torch.long)
    chain = recover_aligned_chain(
        candidate_tokens,
        predictions,
        endpoint_raw,
        eos_token_id=model.config.eos_token_id,
        max_states=max_states,
    )
    tokens = candidate_tokens[chain]
    selected_hidden = torch.stack(hidden_states)[chain]
    selected_predictions = predictions[chain]
    expected = torch.cat((tokens[1:], tokens[-1:]), dim=0)
    alignment_mask = eos_mask(expected, model.config.eos_token_id)
    aligned = selected_predictions.eq(expected) | ~alignment_mask
    state_count = len(chain)
    padded_states = torch.zeros(
        max_states, block_size, model.config.hidden_size, dtype=torch.bfloat16
    )
    padded_tokens = torch.zeros(max_states, block_size, dtype=torch.long)
    state_mask = torch.zeros(max_states, dtype=torch.bool)
    padded_states[:state_count] = selected_hidden
    padded_tokens[:state_count] = tokens
    state_mask[:state_count] = True
    endpoint = tokens[-1]
    token_mask = eos_mask(endpoint, model.config.eos_token_id)
    times = torch.full((max_states,), float("nan"), dtype=torch.float32)
    times[:state_count] = rho_time_grid(
        state_count, epsilon=epsilon, terminal=terminal, rho=rho
    )
    record = {
        "states": padded_states,
        "state_mask": state_mask,
        "trajectory_tokens": padded_tokens,
        "endpoint_tokens": endpoint,
        "token_mask": token_mask,
        "time_grid": times,
    }
    metadata = {
        "data_id": example["data_id"],
        "jacobian_itr_id": example["jacobian_itr_id"],
        "prompt_ids": prompt,
        "trajectory_length": state_count,
        "raw_trajectory_length": len(raw_trajectory),
        "candidate_state_count": len(candidate_tokens),
        "discarded_candidate_count": len(candidate_tokens) - state_count,
        "chain_candidate_indices": chain,
        "alignment_tokens": int((aligned & alignment_mask).sum()),
        "alignment_total": int(alignment_mask.sum()),
        "alignment_ok": bool(aligned.all()),
    }
    return record, metadata


class SplitWriter:
    def __init__(self, root: Path, split: str, shard_size: int):
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.records: list[dict[str, torch.Tensor]] = []
        self.metadata: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.count = 0
        self.metadata_file = self.root / "metadata.jsonl"
        self.metadata_handle = self.metadata_file.open("w", encoding="utf-8")

    def add(self, record: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
        self.records.append(record)
        self.metadata.append(metadata)
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        shard_name = f"shard-{len(self.shards):05d}.safetensors"
        tensors = {
            key: torch.stack([record[key] for record in self.records])
            for key in self.records[0]
        }
        write_shard(self.root / shard_name, tensors)
        for metadata in self.metadata:
            self.metadata_handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        count = len(self.records)
        self.shards.append({"file": shard_name, "count": count})
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
            "metadata_file": self.metadata_file.name,
            "metadata_sha256": hashlib.sha256(self.metadata_file.read_bytes()).hexdigest(),
            "shards": self.shards,
        }
        path = self.root / "manifest.json"
        write_manifest(path, manifest)
        return path


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    model_config = config["model"]
    source = resolve_repo_path(config, config["paths"]["trajectory_json"])
    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    if cache_root.exists() and any(cache_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"cache directory is not empty: {cache_root}; pass --overwrite")
    cache_root.mkdir(parents=True, exist_ok=True)
    train_limit = args.train_limit or int(training["train_limit"])
    validation_limit = args.validation_limit or int(training["validation_limit"])
    validation_fraction = validation_limit / (train_limit + validation_limit)
    device = torch.device(args.device)
    attention_backend = (
        args.attention_backend or config["evaluation"]["attention_backend"]
    )

    target_path = resolve_repo_path(config, config["paths"]["target_model"])
    tokenizer = AutoTokenizer.from_pretrained(target_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        target_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.config.hidden_size != int(model_config["hidden_size"]):
        raise ValueError("target hidden size does not match config")

    save_file(
        {"weight": model.lm_head.weight.detach().to(torch.bfloat16).cpu().contiguous()},
        str(cache_root / "lm_head.safetensors"),
    )
    group_counts: Counter[str] = Counter()
    for raw in tqdm(iter_json_array(source), desc="index data-id groups"):
        data_id = raw.get("data_id")
        if data_id is None:
            raise ValueError("trajectory record is missing data_id")
        group_counts[str(data_id)] += 1
    data_split, split_plan = plan_grouped_split(
        dict(group_counts),
        seed=int(training["seed"]),
        validation_fraction=validation_fraction,
        validation_minimum=validation_limit,
    )
    if split_plan["train_records"] < train_limit:
        raise ValueError(f"grouped split has too few training records: {split_plan}")

    writers = {
        split: SplitWriter(cache_root, split, int(training["shard_size"]))
        for split in ("train", "validation")
    }
    limits = {"train": train_limit, "validation": validation_limit}
    accepted = Counter()
    rejected = Counter()
    progress = tqdm(total=train_limit + validation_limit, desc="aligned blocks")
    accepted_indices: set[int] = set()
    accepted_groups = {"train": set(), "validation": set()}
    sampling_rounds = 0
    while not all(accepted[name] >= limit for name, limit in limits.items()):
        sampling_rounds += 1
        round_groups = {"train": set(), "validation": set()}
        round_start = sum(accepted.values())
        for raw_index, raw in enumerate(iter_json_array(source)):
            if raw_index in accepted_indices:
                continue
            data_id = str(raw["data_id"])
            split = data_split[data_id]
            if accepted[split] >= limits[split] or data_id in round_groups[split]:
                continue
            try:
                example = normalize_example(raw, int(model_config["block_size"]))
                record, metadata = encode_trajectory(
                    model,
                    example,
                    block_size=int(model_config["block_size"]),
                    state_batch_size=int(training["state_batch_size"]),
                    max_states=int(model_config["max_trajectory_states"]),
                    epsilon=float(config["time"]["epsilon"]),
                    terminal=float(config["time"]["terminal"]),
                    rho=float(config["time"]["rho"]),
                    device=device,
                )
            except HiddenTokenAlignmentError:
                rejected["hidden_token_alignment"] += 1
                continue
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                rejected[type(error).__name__] += 1
                continue
            if args.strict_alignment and not metadata["alignment_ok"]:
                rejected["hidden_token_alignment"] += 1
                continue
            writers[split].add(record, metadata)
            accepted[split] += 1
            accepted_indices.add(raw_index)
            accepted_groups[split].add(data_id)
            round_groups[split].add(data_id)
            progress.update(1)
            progress.set_postfix(
                round=sampling_rounds,
                train=accepted["train"],
                validation=accepted["validation"],
            )
            if all(accepted[name] >= limit for name, limit in limits.items()):
                break
        if sum(accepted.values()) == round_start:
            raise RuntimeError(
                f"balanced sampling could not fill cache limits: {dict(accepted)}"
            )
    progress.close()

    split_hash = stable_hash(sorted(data_split.items()))
    common = {
        "schema_version": CACHE_SCHEMA,
        "created_unix": time.time(),
        "config_digest": config_digest(config),
        "config": public_config(config),
        "data_split_hash": split_hash,
        "shape": {
            "states": [
                int(model_config["max_trajectory_states"]),
                int(model_config["block_size"]),
                int(model_config["hidden_size"]),
            ]
        },
        "dtype": "bfloat16",
        "attention_backend": attention_backend,
        "strict_alignment": bool(args.strict_alignment),
        "rejected": dict(rejected),
        "split_plan": split_plan,
        "sampling_rounds": sampling_rounds,
        "accepted_groups": {
            split: len(groups) for split, groups in accepted_groups.items()
        },
        "lm_head_file": "../lm_head.safetensors",
        "backbone_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    manifests = {split: str(writer.close(common)) for split, writer in writers.items()}
    summary = {
        "accepted": dict(accepted),
        "rejected": dict(rejected),
        "data_split_hash": split_hash,
        "split_plan": split_plan,
        "sampling_rounds": sampling_rounds,
        "accepted_groups": {
            split: len(groups) for split, groups in accepted_groups.items()
        },
        "manifests": manifests,
    }
    (cache_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
