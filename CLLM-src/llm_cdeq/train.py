from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm

from .cache import HiddenTrajectoryDataset, iter_shard_batches
from .config import config_digest, load_config, public_config, resolve_repo_path
from .model import (
    CHECKPOINT_SCHEMA,
    AdapterMetrics,
    CDEQAdapter,
    make_ema,
    trainable_parameter_count,
    update_ema_,
)
from .runtime import move_batch, seed_everything, stable_hash
from .time import sample_continuous_pair


LEGACY_MLP_ONLY_DIAGNOSTIC = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CDEQ-Jacobi adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init", type=int, choices=(0, 1), required=True)
    parser.add_argument("--ct", type=int, choices=(0, 1), required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--init-learning-rate", type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--token-ce-weight", type=float)
    parser.add_argument("--local-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init-steps", type=int)
    parser.add_argument("--ct-q", type=float)
    parser.add_argument("--ct-d", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--overfit", action="store_true", help="validate on the training subset")
    return parser.parse_args()


def build_adapter(config: Mapping[str, Any], use_initializer: bool, rank: int | None = None):
    model = config["model"]
    return CDEQAdapter(
        hidden_size=int(model["hidden_size"]),
        rank=int(rank or model["bottleneck_rank"]),
        multiplier=int(model["mlp_multiplier"]),
        terminal=float(config["time"]["terminal"]),
        use_initializer=use_initializer,
    )


def _select_discrete(
    states: torch.Tensor,
    state_mask: torch.Tensor,
    time_grid: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = state_mask.sum(dim=1)
    if bool(torch.any(counts < 2)):
        raise ValueError("every trajectory must contain at least two valid states")
    later = (
        torch.rand(states.shape[0], device=states.device, generator=generator)
        * (counts - 1)
    ).long() + 1
    batch_index = torch.arange(states.shape[0], device=states.device)
    return (
        states[batch_index, later],
        states[batch_index, later - 1],
        time_grid[batch_index, later],
        time_grid[batch_index, later - 1],
    )


def _interpolate_batch(
    states: torch.Tensor,
    state_mask: torch.Tensor,
    time_grid: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    counts = state_mask.sum(dim=1)
    searchable = torch.where(
        state_mask, time_grid, torch.full_like(time_grid, float("inf"))
    )
    right = torch.searchsorted(searchable, query[:, None]).squeeze(1)
    right = torch.maximum(right, torch.ones_like(right))
    right = torch.minimum(right, counts - 1)
    left = right - 1
    batch_index = torch.arange(states.shape[0], device=states.device)
    left_time = time_grid[batch_index, left]
    right_time = time_grid[batch_index, right]
    weight = (query - left_time) / (right_time - left_time).clamp_min(
        torch.finfo(time_grid.dtype).eps
    )
    return states[batch_index, left] * (1 - weight[:, None, None]) + states[
        batch_index, right
    ] * weight[:, None, None]


def _select_continuous(
    states: torch.Tensor,
    state_mask: torch.Tensor,
    time_grid: torch.Tensor,
    global_step: int,
    time_config: Mapping[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    later_times, earlier_times = sample_continuous_pair(
        states.shape[0],
        global_step,
        epsilon=float(time_config["epsilon"]),
        terminal=float(time_config["terminal"]),
        q=float(time_config["q"]),
        d=int(time_config["d"]),
        k=float(time_config["k"]),
        b=float(time_config["b"]),
        device=states.device,
        dtype=states.dtype,
        generator=generator,
    )
    return (
        _interpolate_batch(states, state_mask, time_grid, later_times),
        _interpolate_batch(states, state_mask, time_grid, earlier_times),
        later_times,
        earlier_times,
    )


def _masked_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _token_ce(
    prediction: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(prediction, lm_head_weight)
    return F.cross_entropy(logits[mask], target_tokens[mask])


def train_step(
    adapter: CDEQAdapter,
    ema: CDEQAdapter,
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    use_ct: bool,
    global_step: int,
    generator: torch.Generator,
    lm_head_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    states = batch["states"]
    state_mask = batch["state_mask"]
    times = batch["time_grid"]
    endpoint = states[
        torch.arange(states.shape[0], device=states.device), state_mask.sum(dim=1) - 1
    ]
    token_mask = batch["token_mask"]
    if use_ct:
        later, earlier, later_t, earlier_t = _select_continuous(
            states, state_mask, times, global_step, config["time"], generator
        )
    else:
        later, earlier, later_t, earlier_t = _select_discrete(
            states, state_mask, times, generator
        )

    with torch.no_grad():
        adjacent_target = ema.consistency(later, later_t)
    adjacent_prediction = adapter.consistency(earlier, earlier_t)
    local_loss = _masked_mse(adjacent_prediction, adjacent_target, token_mask)
    endpoint_prediction = adapter.consistency(earlier, earlier_t)
    endpoint_loss = _masked_smooth_l1(endpoint_prediction, endpoint, token_mask)

    anchor_prediction = None
    if adapter.use_initializer:
        initialized = adapter.initialize(states[:, 0]).detach()
        anchor_prediction = adapter.consistency(
            initialized,
            torch.zeros(states.shape[0], device=states.device, dtype=states.dtype),
        )
        anchor_loss = _masked_smooth_l1(anchor_prediction, endpoint, token_mask)
        endpoint_loss = 0.5 * (endpoint_loss + anchor_loss)

    training = config["training"]
    loss = float(training["local_weight"]) * local_loss + float(
        training["endpoint_weight"]
    ) * endpoint_loss
    token_loss = torch.zeros((), device=states.device)
    token_ce_weight = float(training.get("token_ce_weight", 0.0))
    if token_ce_weight:
        if lm_head_weight is None:
            raise ValueError("token CE was requested but no cached LM head was loaded")
        token_loss = _token_ce(
            endpoint_prediction, batch["endpoint_tokens"], token_mask, lm_head_weight
        )
        if anchor_prediction is not None:
            token_loss = 0.5 * (
                token_loss
                + _token_ce(
                    anchor_prediction, batch["endpoint_tokens"], token_mask, lm_head_weight
                )
            )
        loss = loss + token_ce_weight * token_loss
    return loss, {
        "local": float(local_loss.detach()),
        "endpoint": float(endpoint_loss.detach()),
        "token_ce": float(token_loss.detach()),
    }


def train_initializer(
    adapter: CDEQAdapter,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    steps: int,
) -> float:
    if not adapter.use_initializer:
        return 0.0
    states = batch["states"]
    state_mask = batch["state_mask"]
    endpoint = states[
        torch.arange(states.shape[0], device=states.device), state_mask.sum(dim=1) - 1
    ]
    loss_value = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = adapter.initialize(states[:, 0])
        loss = _masked_smooth_l1(prediction, endpoint.detach(), batch["token_mask"])
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value


@torch.inference_mode()
def evaluate_cache(
    adapter: CDEQAdapter | None,
    dataset: HiddenTrajectoryDataset,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    limit: int | None = None,
) -> AdapterMetrics:
    if adapter is not None:
        adapter.eval()
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
        endpoint = states[
            torch.arange(states.shape[0], device=device), state_mask.sum(dim=1) - 1
        ]
        if adapter is None:
            prediction = states[:, 0]
        else:
            initial = adapter.initialize(states[:, 0]) if adapter.use_initializer else states[:, 0]
            prediction = adapter.consistency(
                initial, torch.zeros(states.shape[0], device=device, dtype=states.dtype)
            )
        mask = batch["token_mask"]
        relative = (
            ((prediction - endpoint) * mask.unsqueeze(-1)).flatten(1).norm(dim=1)
            / (endpoint * mask.unsqueeze(-1)).flatten(1).norm(dim=1).clamp_min(1e-8)
        )
        predicted_tokens = F.linear(prediction, lm_head_weight).argmax(dim=-1)
        correct = predicted_tokens.eq(batch["endpoint_tokens"]) & mask
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


def save_checkpoint(
    path: str | Path,
    adapter: CDEQAdapter,
    ema: CDEQAdapter,
    optimizer: torch.optim.Optimizer,
    init_optimizer: torch.optim.Optimizer | None,
    config: Mapping[str, Any],
    metrics: AdapterMetrics,
    *,
    epoch: int,
    global_step: int,
    use_ct: bool,
    train_manifest: Mapping[str, Any],
) -> None:
    package = {
        "schema_version": CHECKPOINT_SCHEMA,
        "created_unix": time.time(),
        "adapter_state": adapter.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "initializer_optimizer_state": init_optimizer.state_dict() if init_optimizer else None,
        "config": public_config(config),
        "config_digest": config_digest(config),
        "data_split_hash": train_manifest["data_split_hash"],
        "upstream": dict(config["upstream"]),
        "trainable_parameter_count": trainable_parameter_count(adapter),
        "backbone_parameter_count": int(train_manifest["backbone_parameter_count"]),
        "trainable_fraction": trainable_parameter_count(adapter)
        / int(train_manifest["backbone_parameter_count"]),
        "best_validation_metrics": metrics.__dict__,
        "epoch": epoch,
        "global_step": global_step,
        "use_initializer": adapter.use_initializer,
        "use_continuous_time": use_ct,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, path)


def load_checkpoint(
    path: str | Path,
    adapter: CDEQAdapter,
    ema: CDEQAdapter | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    init_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {package.get('schema_version')!r}")
    adapter.load_state_dict(package["adapter_state"])
    if ema is not None:
        ema.load_state_dict(package["ema_state"])
    if optimizer is not None:
        optimizer.load_state_dict(package["optimizer_state"])
    if init_optimizer is not None and package.get("initializer_optimizer_state"):
        init_optimizer.load_state_dict(package["initializer_optimizer_state"])
    return package


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = config["training"]
    for argument, key in (
        (args.epochs, "epochs"),
        (args.learning_rate, "learning_rate"),
        (args.init_learning_rate, "init_learning_rate"),
        (args.token_ce_weight, "token_ce_weight"),
    ):
        if argument is not None:
            training[key] = argument
    if args.local_weight is not None:
        training["local_weight"] = args.local_weight
        training["endpoint_weight"] = 1 - args.local_weight
    if args.rank is not None:
        config["model"]["bottleneck_rank"] = args.rank
    if args.seed is not None:
        training["seed"] = args.seed
    if args.init_steps is not None:
        training["init_steps"] = args.init_steps
    if args.ct_q is not None:
        config["time"]["q"] = args.ct_q
    if args.ct_d is not None:
        config["time"]["d"] = args.ct_d


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _apply_cli_overrides(config, args)
    training = config["training"]
    seed_everything(int(training["seed"]))
    device = torch.device(args.device)
    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    train_data = HiddenTrajectoryDataset(cache_root / "train" / "manifest.json")
    validation_data = (
        train_data
        if args.overfit
        else HiddenTrajectoryDataset(cache_root / "validation" / "manifest.json")
    )
    if train_data.manifest["data_split_hash"] != validation_data.manifest["data_split_hash"]:
        raise ValueError("train and validation manifests use different data splits")
    adapter = build_adapter(config, bool(args.init)).to(device=device, dtype=torch.float32)
    ema = make_ema(adapter)
    optimizer = torch.optim.AdamW(
        adapter.consistency_parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    init_optimizer = None
    if adapter.use_initializer:
        init_optimizer = torch.optim.AdamW(
            adapter.initializer_parameters(),
            lr=float(training["init_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    lm_head = load_file(str(cache_root / "lm_head.safetensors"), device=str(device))["weight"]
    lm_head = lm_head.to(dtype=torch.float32)
    lm_head.requires_grad_(False)

    output_root = Path(args.output_dir) if args.output_dir else resolve_repo_path(
        config, config["paths"]["output_dir"]
    )
    variant = f"init{args.init}_ct{args.ct}_seed{training['seed']}"
    if args.overfit:
        variant += "_overfit"
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    start_epoch = 0
    global_step = 0
    best_error = math.inf
    if args.resume:
        package = load_checkpoint(
            args.resume,
            adapter,
            ema,
            optimizer=optimizer,
            init_optimizer=init_optimizer,
        )
        start_epoch = int(package["epoch"]) + 1
        global_step = int(package["global_step"])
        best_error = float(
            package["best_validation_metrics"]["endpoint_relative_error"]
        )

    generator = torch.Generator(device=device).manual_seed(int(training["seed"]))
    bad_epochs = 0
    train_limit = args.train_limit or len(train_data)
    validation_limit = args.validation_limit or len(validation_data)
    accumulation = int(training["gradient_accumulation"])
    initial_metrics = None
    identity_metrics = None
    initial_metrics_path = output_dir / "initial_metrics.json"
    identity_metrics_path = output_dir / "identity_metrics.json"
    if start_epoch == 0:
        identity_metrics = evaluate_cache(
            None,
            validation_data,
            lm_head,
            device=device,
            batch_size=int(training["batch_size"]),
            limit=validation_limit,
        )
        identity_metrics_path.write_text(
            json.dumps(identity_metrics.__dict__, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"identity_validation": identity_metrics.__dict__}, sort_keys=True))
        initial_metrics = evaluate_cache(
            adapter,
            validation_data,
            lm_head,
            device=device,
            batch_size=int(training["batch_size"]),
            limit=validation_limit,
        )
        initial_metrics_path.write_text(
            json.dumps(initial_metrics.__dict__, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"initial_validation": initial_metrics.__dict__}, sort_keys=True))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(training["epochs"])):
        adapter.train()
        loss_total = 0.0
        examples = 0
        optimizer_steps = 0
        progress = tqdm(
            iter_shard_batches(
                train_data,
                int(training["batch_size"]),
                shuffle=True,
                generator=torch.Generator().manual_seed(int(training["seed"]) + epoch),
            ),
            desc=f"epoch {epoch + 1}",
        )
        for micro_step, batch in enumerate(progress):
            if examples >= train_limit:
                break
            take = min(next(iter(batch.values())).shape[0], train_limit - examples)
            batch = {name: value[:take] for name, value in batch.items()}
            batch = move_batch(batch, device, floating_dtype=torch.float32)
            init_loss = 0.0
            if init_optimizer is not None:
                init_loss = train_initializer(
                    adapter, init_optimizer, batch, int(training["init_steps"])
                )
            loss, parts = train_step(
                adapter,
                ema,
                batch,
                config,
                use_ct=bool(args.ct),
                global_step=global_step,
                generator=generator,
                lm_head_weight=lm_head if float(training["token_ce_weight"]) else None,
            )
            (loss / accumulation).backward()
            if (micro_step + 1) % accumulation == 0 or examples + take >= train_limit:
                torch.nn.utils.clip_grad_norm_(list(adapter.consistency_parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema_(ema, adapter, float(training["ema_decay"]))
                if adapter.initializer is not None:
                    ema.initializer.load_state_dict(adapter.initializer.state_dict())
                global_step += 1
                optimizer_steps += 1
            loss_total += float(loss.detach()) * take
            examples += take
            progress.set_postfix(
                loss=f"{loss_total / examples:.4g}",
                init=f"{init_loss:.4g}",
                endpoint=f"{parts['endpoint']:.4g}",
            )

        metrics = evaluate_cache(
            adapter,
            validation_data,
            lm_head,
            device=device,
            batch_size=int(training["batch_size"]),
            limit=validation_limit,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_total / max(examples, 1),
            "optimizer_steps": optimizer_steps,
            **metrics.__dict__,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))
        save_checkpoint(
            output_dir / "last.pt",
            adapter,
            ema,
            optimizer,
            init_optimizer,
            config,
            metrics,
            epoch=epoch,
            global_step=global_step,
            use_ct=bool(args.ct),
            train_manifest=train_data.manifest,
        )
        if metrics.endpoint_relative_error < best_error:
            best_error = metrics.endpoint_relative_error
            bad_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                adapter,
                ema,
                optimizer,
                init_optimizer,
                config,
                metrics,
                epoch=epoch,
                global_step=global_step,
                use_ct=bool(args.ct),
                train_manifest=train_data.manifest,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(training["patience"]):
                break

    backbone_count = int(train_data.manifest["backbone_parameter_count"])
    run_manifest = {
        "variant": variant,
        "config": public_config(config),
        "config_digest": config_digest(config),
        "trainable_parameter_count": trainable_parameter_count(adapter),
        "backbone_parameter_count": backbone_count,
        "trainable_fraction": trainable_parameter_count(adapter) / backbone_count,
        "data_split_hash": train_data.manifest["data_split_hash"],
        "history_sha256": stable_hash(history_path.read_text(encoding="utf-8")),
        "initial_validation_metrics": (
            initial_metrics.__dict__
            if initial_metrics is not None
            else (
                json.loads(initial_metrics_path.read_text(encoding="utf-8"))
                if initial_metrics_path.exists()
                else None
            )
        ),
        "identity_validation_metrics": (
            identity_metrics.__dict__
            if identity_metrics is not None
            else (
                json.loads(identity_metrics_path.read_text(encoding="utf-8"))
                if identity_metrics_path.exists()
                else None
            )
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
