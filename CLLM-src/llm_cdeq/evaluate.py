from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import HiddenTrajectoryDataset, iter_shard_batches
from .config import load_config, resolve_repo_path
from .model import AdapterMetrics
from .runtime import move_batch, seed_everything
from .train import build_adapter, load_checkpoint


LEGACY_MLP_ONLY_DIAGNOSTIC = True


PROMPT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a one-step CDEQ-Jacobi adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("cache", "gsm8k", "both"), default="cache")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--weights", choices=("ema", "online"), default="ema")
    parser.add_argument("--output-file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"))
    return parser.parse_args()


def load_adapter(checkpoint_path: str | Path, device: torch.device, weights: str):
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = package["config"]
    adapter = build_adapter(
        checkpoint_config,
        bool(package["use_initializer"]),
        rank=int(checkpoint_config["model"]["bottleneck_rank"]),
    )
    load_checkpoint(checkpoint_path, adapter)
    if weights == "ema":
        adapter.load_state_dict(package["ema_state"])
    return adapter.to(device=device, dtype=torch.float32).eval(), package


@torch.inference_mode()
def identity_metrics(
    dataset: HiddenTrajectoryDataset,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    limit: int | None,
) -> AdapterMetrics:
    error_sum = 0.0
    agreement = 0
    token_count = 0
    exact = 0
    examples = 0
    for batch in iter_shard_batches(dataset, batch_size, shuffle=False):
        if limit is not None and examples >= limit:
            break
        if limit is not None:
            take = min(next(iter(batch.values())).shape[0], limit - examples)
            batch = {name: value[:take] for name, value in batch.items()}
        batch = move_batch(batch, device, floating_dtype=torch.float32)
        states, state_mask = batch["states"], batch["state_mask"]
        prediction = states[:, 0]
        endpoint = states[
            torch.arange(states.shape[0], device=device), state_mask.sum(dim=1) - 1
        ]
        mask = batch["token_mask"]
        relative = (
            ((prediction - endpoint) * mask.unsqueeze(-1)).flatten(1).norm(dim=1)
            / (endpoint * mask.unsqueeze(-1)).flatten(1).norm(dim=1).clamp_min(1e-8)
        )
        tokens = F.linear(prediction, lm_head_weight).argmax(dim=-1)
        correct = tokens.eq(batch["endpoint_tokens"]) & mask
        error_sum += float(relative.sum())
        agreement += int(correct.sum())
        token_count += int(mask.sum())
        exact += int((correct | ~mask).all(dim=1).sum())
        examples += states.shape[0]
    return AdapterMetrics(
        endpoint_relative_error=error_sum / max(examples, 1),
        token_agreement=agreement / max(token_count, 1),
        exact_block_match=exact / max(examples, 1),
        examples=examples,
    )


@torch.inference_mode()
def adapter_metrics(
    adapter,
    dataset: HiddenTrajectoryDataset,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    limit: int | None,
) -> AdapterMetrics:
    from .train import evaluate_cache

    return evaluate_cache(
        adapter,
        dataset,
        lm_head_weight,
        device=device,
        batch_size=batch_size,
        limit=limit,
    )


@torch.inference_mode()
def generate_cdeq(
    model,
    tokenizer,
    adapter,
    prompt: str,
    *,
    block_size: int,
    max_new_tokens: int,
    rng: random.Random,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated: list[torch.Tensor] = []
    adapter_seconds = 0.0
    target_seconds = 0.0
    target_start = time.perf_counter()
    prefill = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_key_values = prefill.past_key_values
    prefix_last_hidden = prefill.last_hidden_state[:, -1:]
    first_token = model.lm_head(prefix_last_hidden).argmax(dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    target_seconds += time.perf_counter() - target_start
    population = input_ids[0].tolist()
    eos_reached = False
    while sum(block.numel() for block in generated) < max_new_tokens:
        random_tokens = torch.tensor(
            rng.choices(population, k=block_size - 1), device=device, dtype=torch.long
        ).view(1, -1)
        initial_tokens = torch.cat((first_token, random_tokens), dim=1)
        target_start = time.perf_counter()
        represented = model.model(
            input_ids=initial_tokens,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        shifted = torch.cat((prefix_last_hidden, represented.last_hidden_state[:, :-1]), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        target_seconds += time.perf_counter() - target_start

        adapter_start = time.perf_counter()
        state = shifted.float()
        if adapter.use_initializer:
            state = adapter.initialize(state)
        endpoint_hidden = adapter.consistency(
            state, torch.zeros(1, device=device, dtype=state.dtype)
        )
        endpoint_tokens = model.lm_head(endpoint_hidden.to(model.dtype)).argmax(dim=-1)[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        adapter_seconds += time.perf_counter() - adapter_start
        eos = torch.where(endpoint_tokens.eq(tokenizer.eos_token_id))[0]
        if len(eos):
            eos_reached = True
            generated.append(endpoint_tokens[: int(eos[0]) + 1].cpu())
            break
        generated.append(endpoint_tokens.cpu())
        population.extend(endpoint_tokens.tolist())
        target_start = time.perf_counter()
        committed = model.model(
            input_ids=endpoint_tokens.view(1, -1),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = committed.past_key_values
        prefix_last_hidden = committed.last_hidden_state[:, -1:]
        first_token = model.lm_head(prefix_last_hidden).argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        target_seconds += time.perf_counter() - target_start
    output = torch.cat(generated)[:max_new_tokens]
    longest_repeat = 1
    current_repeat = 1
    for index in range(1, output.numel()):
        if output[index] == output[index - 1]:
            current_repeat += 1
            longest_repeat = max(longest_repeat, current_repeat)
        else:
            current_repeat = 1
    return output, {
        "adapter_seconds": adapter_seconds,
        "target_seconds": target_seconds,
        "generated_tokens": int(output.numel()),
        "eos_reached_samples": int(eos_reached),
        "eos_first_token_samples": int(
            output.numel() > 0 and output[0] == tokenizer.eos_token_id
        ),
        "repetitive_samples": int(longest_repeat >= 10),
    }


@torch.inference_mode()
def backbone_parameter_checksum(model) -> str:
    """Hash full-tensor reductions without transferring the 7B state dict to CPU."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(f"{float(value.sum(dtype=torch.float64)):.17g}".encode("ascii"))
        digest.update(f"{float(value.abs().sum(dtype=torch.float64)):.17g}".encode("ascii"))
    return digest.hexdigest()


def load_gsm8k_questions(path: Path, limit: int | None) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["source"].lower() == "gsm8k":
                questions.append(record["question"])
                if limit is not None and len(questions) >= limit:
                    break
    return questions


def score_official(repo_root: Path, predictions: Path, target_model: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "acc.py",
        "--model_dir",
        str(target_model),
        "--output_file_name",
        str(predictions),
        "--dev_set",
        "gsm8k",
        "--eval_only",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root / "eval" / "gsm8k",
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"num_q\s+(\d+)\s+correct\s+(\d+)\s+ratio\s+([0-9.]+)", result.stdout)
    return {
        "stdout": result.stdout,
        "examples": int(match.group(1)) if match else None,
        "correct": int(match.group(2)) if match else None,
        "accuracy": float(match.group(3)) if match else None,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    adapter, package = load_adapter(args.checkpoint, device, args.weights)
    if package["upstream"] != config["upstream"]:
        raise ValueError("checkpoint and evaluation config use different upstream revisions")
    output_root = resolve_repo_path(config, config["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": args.weights,
        "trainable_parameter_count": package["trainable_parameter_count"],
    }

    if args.mode in ("cache", "both"):
        cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
        dataset = HiddenTrajectoryDataset(cache_root / args.split / "manifest.json")
        if dataset.manifest["data_split_hash"] != package["data_split_hash"]:
            raise ValueError("checkpoint/cache split hash mismatch")
        lm_head = load_file(str(cache_root / "lm_head.safetensors"), device=str(device))["weight"]
        lm_head = lm_head.float()
        limit = args.sample_limit
        identity = identity_metrics(
            dataset,
            lm_head,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            limit=limit,
        )
        adapted = adapter_metrics(
            adapter,
            dataset,
            lm_head,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            limit=limit,
        )
        report["cache"] = {"identity": asdict(identity), "adapter": asdict(adapted)}

    if args.mode in ("gsm8k", "both"):
        repo_root = Path(__file__).resolve().parents[1]
        target_path = resolve_repo_path(config, config["paths"]["target_model"])
        attention_backend = args.attention_backend or config["evaluation"]["attention_backend"]
        model = AutoModelForCausalLM.from_pretrained(
            target_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attention_backend,
        ).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checksum_before = backbone_parameter_checksum(model)
        tokenizer = AutoTokenizer.from_pretrained(target_path, use_fast=False)
        test_path = resolve_repo_path(config, config["paths"]["gsm8k_test"])
        limit = args.sample_limit or int(config["evaluation"]["sample_limit"])
        questions = load_gsm8k_questions(test_path, limit)
        output_file = Path(args.output_file) if args.output_file else output_root / "cdeq_gsm8k.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        timing = {
            "adapter_seconds": 0.0,
            "target_seconds": 0.0,
            "generated_tokens": 0,
            "eos_reached_samples": 0,
            "eos_first_token_samples": 0,
            "repetitive_samples": 0,
            "empty_decoded_samples": 0,
        }
        rng = random.Random(seed)
        with output_file.open("w", encoding="utf-8") as handle:
            for index, question in enumerate(tqdm(questions, desc="GSM8K CDEQ")):
                prompt = PROMPT.format(input=question)
                tokens, sample_timing = generate_cdeq(
                    model,
                    tokenizer,
                    adapter,
                    prompt,
                    block_size=int(config["model"]["block_size"]),
                    max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
                    rng=rng,
                )
                response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                timing["empty_decoded_samples"] += int(not response)
                handle.write(
                    json.dumps(
                        {"id": index, "prompt": prompt, "response": response},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for key, value in sample_timing.items():
                    timing[key] += value
        score = score_official(repo_root, output_file, target_path)
        checksum_after = backbone_parameter_checksum(model)
        if checksum_before != checksum_after:
            raise RuntimeError("frozen backbone checksum changed during evaluation")
        timing["adapter_fraction"] = timing["adapter_seconds"] / max(
            timing["adapter_seconds"] + timing["target_seconds"], 1e-12
        )
        report["gsm8k"] = {
            "prediction_file": str(output_file),
            "timing": timing,
            "score": score,
            "attention_backend": attention_backend,
            "backbone_checksum_before": checksum_before,
            "backbone_checksum_after": checksum_after,
            "backbone_unchanged": True,
        }

    report_path = output_root / f"evaluation_{Path(args.checkpoint).stem}_{args.mode}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
