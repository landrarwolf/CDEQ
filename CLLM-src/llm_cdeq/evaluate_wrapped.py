from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .cllm_cache import OfficialCLLMTrajectoryDataset
from .cllm_step import OfficialCLLMSingleStep
from .config import config_digest, load_config, resolve_repo_path
from .train_wrapped import (
    WRAPPED_CHECKPOINT_SCHEMA,
    build_corrector,
    evaluate_wrapped_cache,
    gate_results,
)
from .wrapped_model import WrappedCLLM


PROMPT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the active official-CLLM wrapped CDEQ checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("cache", "gsm8k", "both"), default="both")
    parser.add_argument("--weights", choices=("online", "ema", "both"), default="both")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"))
    return parser.parse_args()


def load_correctors(
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    *,
    weights: str,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if package.get("schema_version") != WRAPPED_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported wrapped checkpoint schema: {package.get('schema_version')!r}"
        )
    if package.get("config_digest") != config_digest(config):
        raise ValueError("wrapped checkpoint and evaluation config digests do not match")
    names = ("online", "ema") if weights == "both" else (weights,)
    correctors: dict[str, torch.nn.Module] = {}
    for name in names:
        corrector = build_corrector(config)
        state_key = "corrector_state" if name == "online" else "ema_state"
        corrector.load_state_dict(package[state_key])
        correctors[name] = corrector.to(device=device, dtype=torch.float32).eval()
    return correctors, package


def initial_block(
    first_token: torch.Tensor,
    prefix_ids: torch.Tensor,
    block_size: int,
    rng: random.Random,
) -> torch.Tensor:
    if block_size < 1:
        raise ValueError("block size must be positive")
    if block_size == 1:
        return first_token
    population = prefix_ids[0].tolist()
    random_tokens = torch.tensor(
        rng.choices(population, k=block_size - 1),
        dtype=torch.long,
        device=prefix_ids.device,
    ).view(1, -1)
    return torch.cat((first_token, random_tokens), dim=1)


@torch.inference_mode()
def refine_block(
    wrapper: WrappedCLLM,
    prefill,
    current_tokens: torch.Tensor,
    *,
    method: str,
    max_rounds: int,
) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
    if method not in ("official_single", "official_fixed", "wrapped"):
        raise ValueError(f"unsupported refinement method: {method}")
    rounds_limit = max_rounds if method == "official_fixed" else 1
    disable_adapter = method != "wrapped"
    cllm_nfe = corrector_nfe = initializer_nfe = 0
    corrector_seconds = 0.0
    converged = False
    rounds = 0
    for round_index in range(rounds_limit):
        output = wrapper(
            prefill,
            current_tokens,
            torch.tensor(0.0, device=current_tokens.device),
            round_index=round_index,
            disable_adapter=disable_adapter,
        )
        rounds += 1
        cllm_nfe += output.cllm_backbone_nfe
        corrector_nfe += output.corrector_nfe
        initializer_nfe += output.initializer_nfe
        corrector_seconds += output.corrector_latency_seconds
        if torch.equal(output.tokens, current_tokens):
            current_tokens = output.tokens
            converged = True
            break
        current_tokens = output.tokens
    return current_tokens, {
        "rounds": rounds,
        "cllm_backbone_nfe": cllm_nfe,
        "corrector_nfe": corrector_nfe,
        "initializer_nfe": initializer_nfe,
        "corrector_seconds": corrector_seconds,
        "converged": converged,
    }


@torch.inference_mode()
def generate_answer(
    wrapper: WrappedCLLM,
    tokenizer,
    prompt: str,
    *,
    method: str,
    block_size: int,
    max_new_tokens: int,
    max_rounds: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
    if max_new_tokens < 1 or max_rounds < 1:
        raise ValueError("generation limits must be positive")
    device = next(wrapper.cllm_step.model.parameters()).device
    prefix_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    rng = random.Random(seed)
    generated: list[torch.Tensor] = []
    totals: dict[str, int | float] = {
        "blocks": 0,
        "prompt_prefill_nfe": 0,
        "rounds": 0,
        "cllm_backbone_nfe": 0,
        "corrector_nfe": 0,
        "initializer_nfe": 0,
        "corrector_seconds": 0.0,
        "converged_blocks": 0,
    }
    eos_reached = False
    repeated_block = False
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(math.ceil(max_new_tokens / block_size)):
        # ponytail: full re-prefill is correctness-only; add committed-block KV reuse
        # before using this path for a speed claim.
        prefill = wrapper.prefill(prefix_ids)
        totals["prompt_prefill_nfe"] += prefill.prompt_nfe
        current = initial_block(prefill.first_token, prefix_ids, block_size, rng)
        tokens, block_metrics = refine_block(
            wrapper,
            prefill,
            current,
            method=method,
            max_rounds=max_rounds,
        )
        totals["blocks"] += 1
        for key in (
            "rounds",
            "cllm_backbone_nfe",
            "corrector_nfe",
            "initializer_nfe",
            "corrector_seconds",
        ):
            totals[key] += block_metrics[key]
        totals["converged_blocks"] += int(block_metrics["converged"])
        repeated_block = repeated_block or bool(tokens[:, 1:].eq(tokens[:, :1]).all())

        remaining = max_new_tokens - sum(part.numel() for part in generated)
        accepted = tokens[0, :remaining]
        eos = torch.where(accepted.eq(tokenizer.eos_token_id))[0]
        if len(eos):
            accepted = accepted[: int(eos[0]) + 1]
            eos_reached = True
        generated.append(accepted.cpu())
        prefix_ids = torch.cat((prefix_ids, accepted.view(1, -1).to(device)), dim=1)
        if eos_reached or sum(part.numel() for part in generated) >= max_new_tokens:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    output = torch.cat(generated) if generated else torch.empty(0, dtype=torch.long)
    return output, {
        **totals,
        "wall_seconds": wall_seconds,
        "generated_tokens": int(output.numel()),
        "eos_reached": eos_reached,
        "eos_first_token": bool(
            output.numel() and int(output[0]) == int(tokenizer.eos_token_id)
        ),
        "repeated_block": repeated_block,
        "truncated": not eos_reached,
        "all_blocks_converged": totals["converged_blocks"] == totals["blocks"],
    }


def load_gsm8k_questions(path: Path, limit: int) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["source"].lower() == "gsm8k":
                questions.append(record["question"])
                if len(questions) >= limit:
                    break
    if len(questions) != limit:
        raise ValueError(f"requested {limit} GSM8K questions, found {len(questions)}")
    return questions


def score_predictions(repo_root: Path, predictions: Path, model_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "acc.py",
            "--model_dir",
            str(model_path),
            "--output_file_name",
            str(predictions),
            "--dev_set",
            "gsm8k",
            "--eval_only",
        ],
        cwd=repo_root / "eval" / "gsm8k",
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"num_q\s+(\d+)\s+correct\s+(\d+)\s+ratio\s+([0-9.]+)", result.stdout)
    if match is None:
        raise RuntimeError(f"could not parse official GSM8K score:\n{result.stdout}")
    return {
        "examples": int(match.group(1)),
        "correct": int(match.group(2)),
        "accuracy": float(match.group(3)),
        "stdout": result.stdout,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["model"].get("operator") != "official_cllm":
        raise ValueError("evaluate_wrapped requires model.operator=official_cllm")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("sample limit must be positive")
    device = torch.device(args.device)
    correctors, package = load_correctors(
        args.checkpoint, config, weights=args.weights, device=device
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_repo_path(config, config["paths"]["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": WRAPPED_CHECKPOINT_SCHEMA,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": list(correctors),
    }

    if args.mode in ("cache", "both"):
        cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
        dataset = OfficialCLLMTrajectoryDataset(
            cache_root / "validation" / "manifest.json"
        )
        if dataset.manifest["data_split_hash"] != package["data_split_hash"]:
            raise ValueError("wrapped checkpoint and validation cache split hashes differ")
        lm_head_weight = load_file(
            str(cache_root / "lm_head.safetensors"), device=str(device)
        )["weight"].float()
        cache_report = {}
        for name, corrector in correctors.items():
            metrics = evaluate_wrapped_cache(
                corrector,
                dataset,
                lm_head_weight,
                device=device,
                batch_size=int(config["training"]["batch_size"]),
            )
            cache_report[name] = {
                "metrics": asdict(metrics),
                "gates": gate_results(metrics, config),
            }
        report["cache"] = cache_report

    if args.mode in ("gsm8k", "both"):
        backend = args.attention_backend or config["evaluation"]["attention_backend"]
        model_path = resolve_repo_path(config, config["paths"]["cllm_model"])
        operator = OfficialCLLMSingleStep.from_pretrained(
            model_path,
            block_size=int(config["model"]["block_size"]),
            attention_backend=backend,
            device=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        wrappers = {
            name: WrappedCLLM(operator, corrector).eval()
            for name, corrector in correctors.items()
        }
        reference_wrapper = next(iter(wrappers.values()))
        methods: list[tuple[str, WrappedCLLM, str]] = [
            ("official_single", reference_wrapper, "official_single"),
            ("official_fixed", reference_wrapper, "official_fixed"),
        ]
        methods.extend(
            (f"wrapped_{name}", wrapper, "wrapped")
            for name, wrapper in wrappers.items()
        )
        sample_limit = args.sample_limit or int(config["evaluation"]["sample_limit"])
        max_new_tokens = args.max_new_tokens or int(config["evaluation"]["max_new_tokens"])
        max_rounds = args.max_rounds or int(config["model"]["max_trajectory_states"])
        questions = load_gsm8k_questions(
            resolve_repo_path(config, config["paths"]["gsm8k_test"]), sample_limit
        )
        method_reports = {}
        for method_name, wrapper, refinement_method in methods:
            predictions = output_dir / f"gsm8k_{method_name}_{sample_limit}.jsonl"
            diagnostics = []
            with predictions.open("w", encoding="utf-8") as handle:
                for index, question in enumerate(questions):
                    prompt = PROMPT.format(input=question)
                    tokens, sample_metrics = generate_answer(
                        wrapper,
                        tokenizer,
                        prompt,
                        method=refinement_method,
                        block_size=int(config["model"]["block_size"]),
                        max_new_tokens=max_new_tokens,
                        max_rounds=max_rounds,
                        seed=int(config["training"]["seed"]) + index,
                    )
                    response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    diagnostics.append({**sample_metrics, "empty_decoded": not response})
                    handle.write(
                        json.dumps(
                            {"id": index, "prompt": prompt, "response": response},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "method": method_name,
                                "sample": index + 1,
                                "total": sample_limit,
                                "generated_tokens": sample_metrics["generated_tokens"],
                                "eos_reached": sample_metrics["eos_reached"],
                            },
                            sort_keys=True,
                        )
                    )
            score = score_predictions(Path(__file__).resolve().parents[1], predictions, model_path)
            method_reports[method_name] = {
                "predictions": str(predictions),
                "score": score,
                "diagnostics": diagnostics,
                "empty_outputs": sum(int(row["empty_decoded"]) for row in diagnostics),
                "eos_first_token": sum(int(row["eos_first_token"]) for row in diagnostics),
                "repeated_block": sum(int(row["repeated_block"]) for row in diagnostics),
                "generated_tokens": sum(int(row["generated_tokens"]) for row in diagnostics),
                "cllm_backbone_nfe": sum(int(row["cllm_backbone_nfe"]) for row in diagnostics),
                "corrector_nfe": sum(int(row["corrector_nfe"]) for row in diagnostics),
                "wall_seconds": sum(float(row["wall_seconds"]) for row in diagnostics),
            }
        report["gsm8k"] = {
            "sample_limit": sample_limit,
            "max_new_tokens": max_new_tokens,
            "max_rounds": max_rounds,
            "attention_backend": backend,
            "correctness_only_reprefill": True,
            "methods": method_reports,
        }

    report_path = output_dir / "wrapped_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
