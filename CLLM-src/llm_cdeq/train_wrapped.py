from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .cllm_cache import OfficialCLLMTrajectoryDataset
from .config import config_digest, load_config, public_config, resolve_repo_path
from .corrector import TransformerResidualCorrector, corrector_parameter_count
from .model import update_ema_
from .runtime import move_batch, seed_everything, stable_hash


WRAPPED_CHECKPOINT_SCHEMA = "llm_cdeq_wrapped_checkpoint_v1"


@dataclass(frozen=True)
class WrappedMetrics:
    endpoint_relative_error: float
    baseline_endpoint_relative_error: float
    endpoint_error_improvement: float
    token_agreement: float
    baseline_token_agreement: float
    exact_block_match: float
    baseline_exact_block_match: float
    safe_violation_rate: float
    eos_collapse_rate: float
    repeated_block_rate: float
    examples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short-gate training for official CLLM plus Transformer corrector"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--init", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ct", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_corrector(config: Mapping[str, Any]) -> TransformerResidualCorrector:
    model = config["model"]
    return TransformerResidualCorrector(
        hidden_size=int(model["hidden_size"]),
        rank=int(model["corrector_rank"]),
        block_size=int(model["block_size"]),
        layers=int(model["corrector_layers"]),
        heads=int(model["corrector_heads"]),
        ffn_size=int(model["corrector_ffn_size"]),
        terminal=float(config["time"]["terminal"]),
        preserve_first_position=bool(model.get("preserve_first_position", True)),
    )


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return _masked_mean(
        F.mse_loss(prediction, target, reduction="none").mean(dim=-1), mask
    )


def _masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return _masked_mean(
        F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1), mask
    )


def _samplewise_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    return (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _select_adjacent(
    batch: Mapping[str, torch.Tensor], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = batch["canonical_hidden"]
    state_mask = batch["state_mask"]
    counts = state_mask.sum(dim=1)
    if bool(torch.any(counts < 2)):
        raise ValueError("every official CLLM trajectory needs at least two states")
    earlier_index = (
        torch.rand(states.shape[0], device=states.device, generator=generator)
        * (counts - 1)
    ).long()
    later_index = earlier_index + 1
    batch_index = torch.arange(states.shape[0], device=states.device)
    return (
        states[batch_index, earlier_index],
        states[batch_index, later_index],
        batch["time_grid"][batch_index, earlier_index],
        batch["time_grid"][batch_index, later_index],
    )


def wrapped_train_step(
    corrector: TransformerResidualCorrector,
    ema: TransformerResidualCorrector,
    batch: Mapping[str, torch.Tensor],
    lm_head_weight: torch.Tensor,
    config: Mapping[str, Any],
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    earlier, later, earlier_time, later_time = _select_adjacent(batch, generator)
    mask = batch["eos_mask"]
    endpoint = batch["endpoint_hidden"]
    endpoint_tokens = batch["endpoint_tokens"]
    endpoint_prediction = corrector(earlier, earlier_time)
    with torch.no_grad():
        local_target = ema(earlier, earlier_time)
    local_prediction = corrector(later, later_time)
    local_loss = _masked_mse(local_prediction, local_target, mask)
    endpoint_loss = _masked_smooth_l1(endpoint_prediction, endpoint, mask)
    corrected_sample = _samplewise_smooth_l1(endpoint_prediction, endpoint, mask)
    base_sample = _samplewise_smooth_l1(earlier, endpoint, mask)

    training = config["training"]
    safe_margin = float(training.get("safe_margin", 0.0))
    safe_loss = F.relu(corrected_sample - base_sample + safe_margin).mean()
    logits = F.linear(endpoint_prediction, lm_head_weight)
    token_loss = F.cross_entropy(logits[mask], endpoint_tokens[mask])
    main_loss = float(training["local_weight"]) * local_loss + float(
        training["endpoint_weight"]
    ) * endpoint_loss
    total = (
        main_loss
        + float(training["safe_weight"]) * safe_loss
        + float(training["token_ce_weight"]) * token_loss
    )
    violation_rate = (corrected_sample > base_sample).float().mean()
    return total, {
        "main": float(main_loss.detach()),
        "local": float(local_loss.detach()),
        "endpoint": float(endpoint_loss.detach()),
        "safe": float(safe_loss.detach()),
        "safe_violation_rate": float(violation_rate.detach()),
        "token_ce": float(token_loss.detach()),
    }


@torch.inference_mode()
def evaluate_wrapped_cache(
    corrector: TransformerResidualCorrector,
    dataset: OfficialCLLMTrajectoryDataset,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> WrappedMetrics:
    corrector.eval()
    error_sum = baseline_error_sum = 0.0
    token_correct = baseline_token_correct = token_total = 0
    exact = baseline_exact = violations = eos_collapse = repeated = examples = 0
    for raw_batch in dataset.iter_batches(batch_size, shuffle=False):
        batch = move_batch(raw_batch, device, floating_dtype=torch.float32)
        base = batch["canonical_hidden"][:, 0]
        endpoint = batch["endpoint_hidden"]
        mask = batch["eos_mask"]
        prediction = corrector(
            base,
            torch.zeros(base.shape[0], device=device, dtype=base.dtype),
        )
        masked = mask.unsqueeze(-1)
        denominator = (endpoint * masked).flatten(1).norm(dim=1).clamp_min(1e-8)
        error = ((prediction - endpoint) * masked).flatten(1).norm(dim=1) / denominator
        baseline_error = ((base - endpoint) * masked).flatten(1).norm(dim=1) / denominator
        error_sum += float(error.sum())
        baseline_error_sum += float(baseline_error.sum())
        prediction_tokens = F.linear(prediction, lm_head_weight).argmax(dim=-1)
        baseline_tokens = F.linear(base, lm_head_weight).argmax(dim=-1)
        target_tokens = batch["endpoint_tokens"]
        correct = prediction_tokens.eq(target_tokens) & mask
        baseline_correct = baseline_tokens.eq(target_tokens) & mask
        token_correct += int(correct.sum())
        baseline_token_correct += int(baseline_correct.sum())
        token_total += int(mask.sum())
        exact += int((correct | ~mask).all(dim=1).sum())
        baseline_exact += int((baseline_correct | ~mask).all(dim=1).sum())
        corrected_safe = _samplewise_smooth_l1(prediction, endpoint, mask)
        baseline_safe = _samplewise_smooth_l1(base, endpoint, mask)
        violations += int((corrected_safe > baseline_safe).sum())
        # These are empirical diagnostics, not correctness guarantees.
        eos_token_id = dataset.manifest.get("eos_token_id")
        if eos_token_id is not None:
            eos_collapse += int(
                prediction_tokens.eq(int(eos_token_id)).all(dim=1).sum()
            )
        repeated += int(prediction_tokens[:, 1:].eq(prediction_tokens[:, :1]).all(dim=1).sum())
        examples += base.shape[0]
    mean_error = error_sum / max(examples, 1)
    mean_baseline = baseline_error_sum / max(examples, 1)
    return WrappedMetrics(
        endpoint_relative_error=mean_error,
        baseline_endpoint_relative_error=mean_baseline,
        endpoint_error_improvement=(mean_baseline - mean_error) / max(mean_baseline, 1e-8),
        token_agreement=token_correct / max(token_total, 1),
        baseline_token_agreement=baseline_token_correct / max(token_total, 1),
        exact_block_match=exact / max(examples, 1),
        baseline_exact_block_match=baseline_exact / max(examples, 1),
        safe_violation_rate=violations / max(examples, 1),
        eos_collapse_rate=eos_collapse / max(examples, 1),
        repeated_block_rate=repeated / max(examples, 1),
        examples=examples,
    )


def gate_results(metrics: WrappedMetrics, config: Mapping[str, Any]) -> dict[str, bool]:
    evaluation = config["evaluation"]
    return {
        "endpoint_error_improvement": metrics.endpoint_error_improvement
        >= float(evaluation["endpoint_error_improvement_gate"]),
        "token_agreement_preserved": metrics.token_agreement
        >= metrics.baseline_token_agreement
        - float(evaluation["token_agreement_drop_gate"]),
        "safe_violation_rate": metrics.safe_violation_rate
        <= float(evaluation["safe_violation_rate_gate"]),
        "no_eos_collapse": metrics.eos_collapse_rate == 0,
        "no_repeated_output_collapse": metrics.repeated_block_rate == 0,
    }


def save_wrapped_checkpoint(
    path: Path,
    corrector: TransformerResidualCorrector,
    ema: TransformerResidualCorrector,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    metrics: WrappedMetrics,
    *,
    global_step: int,
) -> None:
    trainable = corrector_parameter_count(corrector)
    backbone = int(manifest["backbone_parameter_count"])
    package = {
        "schema_version": WRAPPED_CHECKPOINT_SCHEMA,
        "created_unix": time.time(),
        "corrector_state": corrector.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": public_config(config),
        "config_digest": config_digest(config),
        "data_split_hash": manifest["data_split_hash"],
        "cache_schema": manifest["schema_version"],
        "operator": manifest["operator"],
        "trainable_parameter_count": trainable,
        "backbone_parameter_count": backbone,
        "trainable_fraction": trainable / backbone,
        "backbone_checksum_before": manifest["backbone_checksum"],
        "backbone_checksum_after": manifest["backbone_checksum"],
        "backbone_checksum_evidence": "frozen official CLLM is read-only and absent from optimizer",
        "best_validation_metrics": asdict(metrics),
        "global_step": global_step,
        "use_initializer": False,
        "use_continuous_time": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, path)


def load_wrapped_checkpoint(
    path: str | Path,
    corrector: TransformerResidualCorrector,
    ema: TransformerResidualCorrector | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("schema_version") != WRAPPED_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported wrapped checkpoint schema: {package.get('schema_version')!r}"
        )
    corrector.load_state_dict(package["corrector_state"])
    if ema is not None:
        ema.load_state_dict(package["ema_state"])
    if optimizer is not None:
        optimizer.load_state_dict(package["optimizer_state"])
    return package


def main() -> None:
    args = parse_args()
    if args.init or args.ct:
        raise ValueError(
            "initializer and CT remain paused until the base wrapped-CLLM 64-block gate passes"
        )
    config = load_config(args.config)
    if config["model"].get("operator") != "official_cllm":
        raise ValueError("train_wrapped requires model.operator=official_cllm")
    seed_everything(int(config["training"]["seed"]))
    device = torch.device(args.device)
    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    dataset = OfficialCLLMTrajectoryDataset(cache_root / "train" / "manifest.json")
    if dataset.manifest["cllm_model_id"] != "cllm/consistency-llm-7b-math":
        raise ValueError("wrapped training cache was not generated by official CLLM")
    if float(dataset.manifest["cache_hidden_token_alignment"]) != 1.0:
        raise ValueError("official CLLM cache alignment is not 100%")

    corrector = build_corrector(config).to(device=device, dtype=torch.float32)
    ema = copy.deepcopy(corrector).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    trainable = corrector_parameter_count(corrector)
    backbone = int(dataset.manifest["backbone_parameter_count"])
    if trainable / backbone >= 0.01:
        raise ValueError("corrector exceeds the 1% backbone parameter gate")
    lm_head_weight = load_file(
        str(cache_root / "lm_head.safetensors"), device=str(device)
    )["weight"].to(torch.float32)
    lm_head_weight.requires_grad_(False)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_steps = min(
        int(args.max_steps or training["max_optimizer_steps"]),
        int(training["max_optimizer_steps"]),
    )
    output_dir = Path(args.output_dir) if args.output_dir else resolve_repo_path(
        config, config["paths"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    generator = torch.Generator(device=device).manual_seed(int(training["seed"]))
    global_step = 0
    best_error = math.inf
    bad_validations = 0
    baseline_metrics = evaluate_wrapped_cache(
        corrector,
        dataset,
        lm_head_weight,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(asdict(baseline_metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"initial": asdict(baseline_metrics)}, sort_keys=True))

    while global_step < max_steps and bad_validations < int(training["patience"]):
        corrector.train()
        epoch_generator = torch.Generator().manual_seed(
            int(training["seed"]) + global_step
        )
        for raw_batch in dataset.iter_batches(
            int(training["batch_size"]), shuffle=True, generator=epoch_generator
        ):
            batch = move_batch(raw_batch, device, floating_dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = wrapped_train_step(
                corrector,
                ema,
                batch,
                lm_head_weight,
                config,
                generator=generator,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("wrapped corrector training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                corrector.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            update_ema_(ema, corrector, float(training["ema_decay"]))
            global_step += 1
            record: dict[str, Any] = {
                "global_step": global_step,
                "loss": float(loss.detach()),
                **parts,
            }
            if global_step % int(training["validation_interval"]) == 0 or global_step == max_steps:
                metrics = evaluate_wrapped_cache(
                    corrector,
                    dataset,
                    lm_head_weight,
                    device=device,
                    batch_size=int(training["batch_size"]),
                )
                record["validation"] = asdict(metrics)
                save_wrapped_checkpoint(
                    output_dir / "last.pt",
                    corrector,
                    ema,
                    optimizer,
                    config,
                    dataset.manifest,
                    metrics,
                    global_step=global_step,
                )
                if metrics.endpoint_relative_error < best_error:
                    best_error = metrics.endpoint_relative_error
                    bad_validations = 0
                    save_wrapped_checkpoint(
                        output_dir / "best.pt",
                        corrector,
                        ema,
                        optimizer,
                        config,
                        dataset.manifest,
                        metrics,
                        global_step=global_step,
                    )
                else:
                    bad_validations += 1
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
            if global_step >= max_steps or bad_validations >= int(training["patience"]):
                break

    package = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    corrector.load_state_dict(package["corrector_state"])
    final_metrics = evaluate_wrapped_cache(
        corrector,
        dataset,
        lm_head_weight,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    gates = gate_results(final_metrics, config)
    report = {
        "schema_version": WRAPPED_CHECKPOINT_SCHEMA,
        "global_step": global_step,
        "stopped_by_patience": bad_validations >= int(training["patience"]),
        "metrics": asdict(final_metrics),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "long_training_allowed": False,
        "trainable_parameter_count": trainable,
        "backbone_parameter_count": backbone,
        "trainable_fraction": trainable / backbone,
        "backbone_checksum_before": dataset.manifest["backbone_checksum"],
        "backbone_checksum_after": dataset.manifest["backbone_checksum"],
        "history_sha256": stable_hash(history_path.read_text(encoding="utf-8")),
    }
    (output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
