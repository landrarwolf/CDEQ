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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--overfit", action="store_true")
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
    epoch: int = -1,
    epoch_complete: bool = True,
    best_endpoint_relative_error: float | None = None,
    best_metrics: WrappedMetrics | None = None,
    ema_metrics: WrappedMetrics | None = None,
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
        "validation_metrics": asdict(metrics),
        "best_validation_metrics": asdict(best_metrics or metrics),
        "ema_validation_metrics": asdict(ema_metrics) if ema_metrics else None,
        "global_step": global_step,
        "epoch": epoch,
        "epoch_complete": epoch_complete,
        "best_endpoint_relative_error": (
            metrics.endpoint_relative_error
            if best_endpoint_relative_error is None
            else best_endpoint_relative_error
        ),
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


def validate_cache_pair(
    train_dataset: OfficialCLLMTrajectoryDataset,
    validation_dataset: OfficialCLLMTrajectoryDataset,
    config: Mapping[str, Any],
) -> None:
    train_manifest = train_dataset.manifest
    validation_manifest = validation_dataset.manifest
    if train_manifest.get("split") != "train":
        raise ValueError("training manifest is not labeled train")
    if validation_manifest.get("split") != "validation":
        raise ValueError("validation manifest is not labeled validation")
    for key in (
        "schema_version",
        "operator",
        "data_split_hash",
        "backbone_checksum",
        "cllm_model_id",
        "cllm_model_revision",
    ):
        if train_manifest.get(key) != validation_manifest.get(key):
            raise ValueError(f"train/validation cache mismatch for {key}")
    if train_manifest.get("config_digest") != config_digest(config):
        raise ValueError("cache and training config digests do not match")
    if int(train_manifest.get("data_id_overlap", -1)) != 0:
        raise ValueError("official CLLM cache reports train/validation leakage")


def prepare_output_dir(path: Path, *, resume: bool) -> None:
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {path}")
        return
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def validate_resume_history(history_path: Path, package: Mapping[str, Any]) -> None:
    if not history_path.is_file():
        raise FileNotFoundError("resume history.jsonl is missing")
    lines = history_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("resume history.jsonl is empty")
    tail = json.loads(lines[-1])
    if (
        int(tail.get("epoch", -1)) != int(package.get("epoch", -2))
        or int(tail.get("global_step", -1)) != int(package.get("global_step", -2))
    ):
        raise ValueError("resume checkpoint does not match the history tail")


def main() -> None:
    args = parse_args()
    if args.init or args.ct:
        raise ValueError(
            "initializer and CT remain paused until the base wrapped-CLLM 64-block gate passes"
        )
    config = load_config(args.config)
    if config["model"].get("operator") != "official_cllm":
        raise ValueError("train_wrapped requires model.operator=official_cllm")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max steps must be positive")
    seed_everything(int(config["training"]["seed"]))
    device = torch.device(args.device)
    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    train_dataset = OfficialCLLMTrajectoryDataset(cache_root / "train" / "manifest.json")
    if args.overfit:
        validation_dataset = train_dataset
    else:
        validation_manifest = cache_root / "validation" / "manifest.json"
        if not validation_manifest.is_file():
            raise FileNotFoundError(
                "wrapped training requires a held-out validation manifest; "
                "use --overfit only for an explicit train-set gate"
            )
        validation_dataset = OfficialCLLMTrajectoryDataset(validation_manifest)
        validate_cache_pair(train_dataset, validation_dataset, config)
    if train_dataset.manifest["cllm_model_id"] != "cllm/consistency-llm-7b-math":
        raise ValueError("wrapped training cache was not generated by official CLLM")
    if float(train_dataset.manifest["cache_hidden_token_alignment"]) != 1.0:
        raise ValueError("official CLLM cache alignment is not 100%")
    if float(validation_dataset.manifest["cache_hidden_token_alignment"]) != 1.0:
        raise ValueError("official CLLM validation cache alignment is not 100%")

    corrector = build_corrector(config).to(device=device, dtype=torch.float32)
    ema = copy.deepcopy(corrector).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    trainable = corrector_parameter_count(corrector)
    backbone = int(train_dataset.manifest["backbone_parameter_count"])
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
    batch_size = int(training["batch_size"])
    steps_per_epoch = sum(
        math.ceil(int(shard["count"]) / batch_size)
        for shard in train_dataset.manifest["shards"]
    )
    configured_epochs = training.get("epochs")
    max_steps = args.max_steps
    if args.epochs is not None:
        epochs = args.epochs
    elif configured_epochs is not None:
        epochs = int(configured_epochs)
    else:
        if max_steps is None:
            max_steps = int(training["max_optimizer_steps"])
        epochs = math.ceil(max_steps / steps_per_epoch)

    resume_path = Path(args.resume).resolve() if args.resume else None
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif resume_path is not None:
        output_dir = resume_path.parent
    else:
        output_dir = resolve_repo_path(config, config["paths"]["output_dir"]).resolve()
    if resume_path is not None and resume_path.parent != output_dir:
        raise ValueError("resume checkpoint and output directory must have the same parent")
    if resume_path is not None and resume_path.name != "last.pt":
        raise ValueError("epoch resume requires the run's last.pt checkpoint")
    prepare_output_dir(output_dir, resume=resume_path is not None)
    history_path = output_dir / "history.jsonl"
    global_step = 0
    best_error = math.inf
    best_metrics: WrappedMetrics | None = None
    start_epoch = 0
    if resume_path is not None:
        package = load_wrapped_checkpoint(
            resume_path, corrector, ema, optimizer=optimizer
        )
        if package.get("config_digest") != config_digest(config):
            raise ValueError("resume checkpoint and training config digests do not match")
        if package.get("data_split_hash") != train_dataset.manifest["data_split_hash"]:
            raise ValueError("resume checkpoint and cache split hashes do not match")
        if not package.get("epoch_complete", False):
            raise ValueError("cannot resume a checkpoint saved during a partial epoch")
        if int(package.get("epoch", -1)) < 0:
            raise ValueError("resume checkpoint does not contain an epoch cursor")
        validate_resume_history(history_path, package)
        start_epoch = int(package["epoch"]) + 1
        global_step = int(package["global_step"])
        best_error = float(package["best_endpoint_relative_error"])
        best_metrics = WrappedMetrics(**package["best_validation_metrics"])
    else:
        baseline_online = evaluate_wrapped_cache(
            corrector,
            validation_dataset,
            lm_head_weight,
            device=device,
            batch_size=batch_size,
        )
        baseline_ema = evaluate_wrapped_cache(
            ema,
            validation_dataset,
            lm_head_weight,
            device=device,
            batch_size=batch_size,
        )
        initial = {"online": asdict(baseline_online), "ema": asdict(baseline_ema)}
        (output_dir / "initial_metrics.json").write_text(
            json.dumps(initial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"initial": initial}, sort_keys=True))

    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        corrector.train()
        epoch_seed = int(training["seed"]) + epoch
        epoch_generator = torch.Generator().manual_seed(epoch_seed)
        pair_generator = torch.Generator(device=device).manual_seed(epoch_seed)
        totals: dict[str, float] = {}
        processed_batches = 0
        for raw_batch in train_dataset.iter_batches(
            batch_size, shuffle=True, generator=epoch_generator
        ):
            if max_steps is not None and global_step >= max_steps:
                break
            batch = move_batch(raw_batch, device, floating_dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = wrapped_train_step(
                corrector,
                ema,
                batch,
                lm_head_weight,
                config,
                generator=pair_generator,
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
            processed_batches += 1
            values = {"loss": float(loss.detach()), **parts}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
        if processed_batches == 0:
            break

        epoch_complete = processed_batches == steps_per_epoch
        online_metrics = evaluate_wrapped_cache(
            corrector,
            validation_dataset,
            lm_head_weight,
            device=device,
            batch_size=batch_size,
        )
        ema_metrics = evaluate_wrapped_cache(
            ema,
            validation_dataset,
            lm_head_weight,
            device=device,
            batch_size=batch_size,
        )
        improved = online_metrics.endpoint_relative_error < best_error
        if improved:
            best_metrics = online_metrics
        best_error = min(best_error, online_metrics.endpoint_relative_error)
        record = {
            "epoch": epoch,
            "epoch_number": epoch + 1,
            "epoch_complete": epoch_complete,
            "global_step": global_step,
            "train": {key: value / processed_batches for key, value in totals.items()},
            "validation": {
                "online": asdict(online_metrics),
                "ema": asdict(ema_metrics),
            },
        }
        save_wrapped_checkpoint(
            output_dir / "last.pt",
            corrector,
            ema,
            optimizer,
            config,
            train_dataset.manifest,
            online_metrics,
            global_step=global_step,
            epoch=epoch,
            epoch_complete=epoch_complete,
            best_endpoint_relative_error=best_error,
            best_metrics=best_metrics,
            ema_metrics=ema_metrics,
        )
        if improved:
            save_wrapped_checkpoint(
                output_dir / "best.pt",
                corrector,
                ema,
                optimizer,
                config,
                train_dataset.manifest,
                online_metrics,
                global_step=global_step,
                epoch=epoch,
                epoch_complete=epoch_complete,
                best_endpoint_relative_error=best_error,
                best_metrics=best_metrics,
                ema_metrics=ema_metrics,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))
        completed_epoch = epoch
        if max_steps is not None and global_step >= max_steps:
            break

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("training did not produce best.pt")
    package = torch.load(best_path, map_location="cpu", weights_only=False)
    corrector.load_state_dict(package["corrector_state"])
    ema.load_state_dict(package["ema_state"])
    final_online = evaluate_wrapped_cache(
        corrector,
        validation_dataset,
        lm_head_weight,
        device=device,
        batch_size=batch_size,
    )
    final_ema = evaluate_wrapped_cache(
        ema,
        validation_dataset,
        lm_head_weight,
        device=device,
        batch_size=batch_size,
    )
    online_gates = gate_results(final_online, config)
    ema_gates = gate_results(final_ema, config)
    report = {
        "schema_version": WRAPPED_CHECKPOINT_SCHEMA,
        "requested_epochs": epochs,
        "completed_epochs": max(completed_epoch + 1, start_epoch),
        "global_step": global_step,
        "overfit": bool(args.overfit),
        "metrics": {"online": asdict(final_online), "ema": asdict(final_ema)},
        "gates": {"online": online_gates, "ema": ema_gates},
        "all_gates_passed": {
            "online": all(online_gates.values()),
            "ema": all(ema_gates.values()),
        },
        "next_phase_allowed": all(online_gates.values()) or all(ema_gates.values()),
        "trainable_parameter_count": trainable,
        "backbone_parameter_count": backbone,
        "trainable_fraction": trainable / backbone,
        "backbone_checksum_before": train_dataset.manifest["backbone_checksum"],
        "backbone_checksum_after": train_dataset.manifest["backbone_checksum"],
        "history_sha256": stable_hash(history_path.read_text(encoding="utf-8")),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
