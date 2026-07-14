from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile AR, Jacobi, CLLM, or CDEQ")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=("ar", "jacobi", "cllm", "cdeq"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"))
    parser.add_argument("--output")
    return parser.parse_args()


def parse_official_speed(stdout: str) -> dict[str, float | None]:
    patterns = {
        "ar_tokens_per_second": r"avg speed of model .* using ar is ([0-9.eE+-]+)",
        "jacobi_tokens_per_second": r"avg speed of model .* using jacobian iteration .* is ([0-9.eE+-]+)",
        "average_convergence_steps": r"average converge steps: ([0-9.eE+-]+)",
    }
    values: dict[str, float | None] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        values[name] = float(match.group(1)) if match else None
    if values["ar_tokens_per_second"] and values["jacobi_tokens_per_second"]:
        values["speedup_over_ar"] = (
            values["jacobi_tokens_per_second"] / values["ar_tokens_per_second"]
        )
    else:
        values["speedup_over_ar"] = None
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    output_root = resolve_repo_path(config, config["paths"]["output_dir"]) / "profiles"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_root / f"{args.method}.json"
    sample_limit = args.sample_limit or int(config["evaluation"]["sample_limit"])
    attention = args.attention_backend or config["evaluation"]["attention_backend"]
    started = time.time()

    if args.method == "cdeq":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for method=cdeq")
        command = [
            sys.executable,
            "-m",
            "llm_cdeq.evaluate",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--mode",
            "gsm8k",
            "--sample-limit",
            str(sample_limit),
            "--attention-backend",
            attention,
        ]
        result = subprocess.run(command, cwd=repo_root, check=True, text=True, capture_output=True)
        evaluation_path = resolve_repo_path(config, config["paths"]["output_dir"]) / (
            f"evaluation_{Path(args.checkpoint).stem}_gsm8k.json"
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        timing = evaluation["gsm8k"]["timing"]
        elapsed = timing["adapter_seconds"] + timing["target_seconds"]
        metrics = {
            "tokens_per_second": timing["generated_tokens"] / max(elapsed, 1e-12),
            "adapter_seconds": timing["adapter_seconds"],
            "target_seconds": timing["target_seconds"],
            "adapter_fraction": timing["adapter_fraction"],
            "accuracy": evaluation["gsm8k"]["score"]["accuracy"],
        }
    else:
        target = resolve_repo_path(config, config["paths"]["target_model"])
        test_model = (
            resolve_repo_path(config, config["paths"]["cllm_model"])
            if args.method == "cllm"
            else target
        )
        command = [
            sys.executable,
            "speedup.py",
            "--filename",
            str(resolve_repo_path(config, config["paths"]["gsm8k_test"])),
            "--test_model_path",
            str(test_model),
            "--teacher_model_path",
            str(target),
            "--max_new_tokens",
            str(config["model"]["block_size"]),
            "--max_new_seq_len",
            str(config["evaluation"]["max_new_tokens"]),
            "--data_size",
            str(sample_limit),
            "--seed",
            str(config["training"]["seed"]),
            "--attention_backend",
            attention,
            "--output_dir",
            str(output_root / f"{args.method}_raw"),
        ]
        result = subprocess.run(
            command,
            cwd=repo_root / "eval" / "gsm8k",
            check=True,
            text=True,
            capture_output=True,
        )
        metrics = parse_official_speed(result.stdout)
        if args.method == "ar":
            metrics["tokens_per_second"] = metrics["ar_tokens_per_second"]
        else:
            metrics["tokens_per_second"] = metrics["jacobi_tokens_per_second"]

    report = {
        "method": args.method,
        "sample_limit": sample_limit,
        "attention_backend": attention,
        "elapsed_wall_seconds": time.time() - started,
        "command": command,
        "metrics": metrics,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
