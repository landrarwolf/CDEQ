from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .cache import HiddenTrajectoryDataset, iter_shard_batches
from .config import load_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CDEQ-Jacobi curves and ablation table")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Repeat for every ablation run directory",
    )
    parser.add_argument("--trajectory-limit", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run must be LABEL=DIR, got {value!r}")
        label, path = value.split("=", 1)
        runs[label] = Path(path)
    return runs


@torch.inference_mode()
def trajectory_curves(config, limit: int, device: torch.device) -> list[dict]:
    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    dataset = HiddenTrajectoryDataset(cache_root / "validation" / "manifest.json")
    lm_head = load_file(str(cache_root / "lm_head.safetensors"), device=str(device))["weight"]
    hidden_error: dict[int, list[float]] = defaultdict(list)
    agreement: dict[int, list[float]] = defaultdict(list)
    seen = 0
    for batch in iter_shard_batches(dataset, 8, shuffle=False):
        if seen >= limit:
            break
        take = min(batch["states"].shape[0], limit - seen)
        states = batch["states"][:take].to(device)
        state_mask = batch["state_mask"][:take].to(device)
        endpoint_tokens = batch["endpoint_tokens"][:take].to(device)
        token_mask = batch["token_mask"][:take].to(device)
        for sample in range(take):
            count = int(state_mask[sample].sum())
            endpoint = states[sample, count - 1].float()
            denominator = (endpoint * token_mask[sample].unsqueeze(-1)).norm().clamp_min(1e-8)
            for step in range(count):
                current = states[sample, step].float()
                error = ((current - endpoint) * token_mask[sample].unsqueeze(-1)).norm() / denominator
                predicted = F.linear(current.to(lm_head.dtype), lm_head).argmax(dim=-1)
                token_score = (
                    (predicted.eq(endpoint_tokens[sample]) & token_mask[sample]).sum()
                    / token_mask[sample].sum().clamp_min(1)
                )
                hidden_error[step].append(float(error))
                agreement[step].append(float(token_score))
        seen += take
    return [
        {
            "step": step,
            "examples": len(hidden_error[step]),
            "endpoint_relative_error": sum(hidden_error[step]) / len(hidden_error[step]),
            "endpoint_token_agreement": sum(agreement[step]) / len(agreement[step]),
        }
        for step in sorted(hidden_error)
    ]


def load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feasibility_summary(rows: list[dict]) -> dict:
    variants = {(row["init"], row["ct"]): row for row in rows}
    required = {(0, 0), (1, 0), (0, 1), (1, 1)}
    if set(variants) != required:
        return {"complete": False, "reason": "requires exactly one row for each Init x CT variant"}
    baseline = variants[(0, 0)]
    baseline_initial_error = baseline.get("initial_endpoint_relative_error")
    baseline_initial_token = baseline.get("initial_token_agreement")
    baseline_gate = {
        "available": baseline_initial_error is not None
        and baseline_initial_error > 0
        and baseline_initial_token is not None
    }
    if baseline_gate["available"]:
        baseline_error_gain = (
            baseline_initial_error - baseline["endpoint_relative_error"]
        ) / baseline_initial_error
        baseline_token_gain = baseline["token_agreement"] - baseline_initial_token
        baseline_gate.update(
            {
                "endpoint_error_relative_gain": baseline_error_gain,
                "token_agreement_absolute_gain": baseline_token_gain,
                "passes": baseline_error_gain >= 0.20 and baseline_token_gain >= 0.05,
            }
        )
    else:
        baseline_gate["passes"] = False

    def improvement(row: dict) -> dict:
        error_gain = (
            baseline["endpoint_relative_error"] - row["endpoint_relative_error"]
        ) / baseline["endpoint_relative_error"]
        token_gain = row["token_agreement"] - baseline["token_agreement"]
        error_degradation = (
            row["endpoint_relative_error"] - baseline["endpoint_relative_error"]
        ) / baseline["endpoint_relative_error"]
        token_degradation = baseline["token_agreement"] - row["token_agreement"]
        passes = (error_gain >= 0.05 and token_degradation <= 0.01) or (
            token_gain >= 0.02 and error_degradation <= 0.01
        )
        return {
            "endpoint_error_relative_gain": error_gain,
            "token_agreement_absolute_gain": token_gain,
            "passes_component_gate": passes,
        }

    init = improvement(variants[(1, 0)])
    ct = improvement(variants[(0, 1)])
    combined = variants[(1, 1)]
    best_error = min(row["endpoint_relative_error"] for row in rows)
    best_token = max(row["token_agreement"] for row in rows)
    combined_best = (
        combined["endpoint_relative_error"] <= best_error
        or combined["token_agreement"] >= best_token
    )
    return {
        "complete": True,
        "baseline_vs_identity": baseline_gate,
        "init_only": init,
        "ct_only": ct,
        "init_ct_best_or_tied_on_primary_metric": combined_best,
        "passes_single_seed_direction_gate": init["passes_component_gate"]
        and ct["passes_component_gate"]
        and combined_best,
        "passes_single_seed_feasibility_gate": baseline_gate["passes"]
        and init["passes_component_gate"]
        and ct["passes_component_gate"]
        and combined_best,
        "note": "Three-seed direction and baseline-vs-identity gates require their matched reruns.",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runs = parse_runs(args.run)
    output_dir = Path(args.output_dir) if args.output_dir else resolve_repo_path(
        config, config["paths"]["output_dir"]
    ) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = trajectory_curves(config, args.trajectory_limit, torch.device(args.device))
    (output_dir / "teacher_trajectory.json").write_text(
        json.dumps(curves, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "teacher_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curves[0].keys())
        writer.writeheader()
        writer.writerows(curves)
    figure, error_axis = plt.subplots(figsize=(6.4, 4.0))
    token_axis = error_axis.twinx()
    steps = [row["step"] for row in curves]
    error_axis.plot(steps, [row["endpoint_relative_error"] for row in curves], "o-", label="hidden error")
    token_axis.plot(steps, [row["endpoint_token_agreement"] for row in curves], "s-", color="tab:orange", label="token agreement")
    error_axis.set(xlabel="Jacobi step", ylabel="Relative hidden error")
    token_axis.set_ylabel("Endpoint token agreement")
    error_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "teacher_trajectory.png", dpi=180)
    plt.close(figure)

    table_rows: list[dict] = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, run_dir in runs.items():
        history = load_history(run_dir / "history.jsonl")
        axes[0].plot(
            [row["epoch"] + 1 for row in history],
            [row["endpoint_relative_error"] for row in history],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [row["epoch"] + 1 for row in history],
            [row["token_agreement"] for row in history],
            marker="o",
            label=label,
        )
        checkpoint = torch.load(
            run_dir / "best.pt", map_location="cpu", weights_only=False
        )
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        initial = run_manifest["initial_validation_metrics"]
        metrics = checkpoint["best_validation_metrics"]
        table_rows.append(
            {
                "method": label,
                "init": int(checkpoint["use_initializer"]),
                "ct": int(checkpoint["use_continuous_time"]),
                "endpoint_relative_error": metrics["endpoint_relative_error"],
                "token_agreement": metrics["token_agreement"],
                "exact_block_match": metrics["exact_block_match"],
                "initial_endpoint_relative_error": initial["endpoint_relative_error"],
                "initial_token_agreement": initial["token_agreement"],
                "trainable_parameters": checkpoint["trainable_parameter_count"],
                "trainable_fraction": checkpoint["trainable_fraction"],
            }
        )
    if runs:
        axes[0].set(xlabel="Epoch", ylabel="Validation endpoint relative error")
        axes[1].set(xlabel="Epoch", ylabel="Validation endpoint token agreement")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "ablation_training_curves.png", dpi=180)
    plt.close(figure)

    if table_rows:
        fields = list(table_rows[0])
        with (output_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table_rows)
        lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in table_rows:
            lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
        (output_dir / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        feasibility = feasibility_summary(table_rows)
        (output_dir / "feasibility.json").write_text(
            json.dumps(feasibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"output_dir": str(output_dir), "runs": list(runs), "trajectory_examples": args.trajectory_limit}, indent=2))


if __name__ == "__main__":
    main()
