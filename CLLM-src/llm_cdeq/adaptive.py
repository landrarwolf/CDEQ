from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F


class AdaptiveAdapter(Protocol):
    """The small part of :class:`CDEQAdapter` needed by Stage A."""

    terminal: float

    def initialize(self, state: torch.Tensor) -> torch.Tensor: ...

    def consistency(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class OracleProjection:
    """Continuous projection of a state onto a padded teacher trajectory."""

    progress: torch.Tensor
    distance: torch.Tensor
    second_distance: torch.Tensor
    margin: torch.Tensor
    segment_index: torch.Tensor
    segment_fraction: torch.Tensor
    projected_state: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class AdaptiveStep:
    """Diagnostics captured after one adapter call."""

    call_index: int
    input_time: torch.Tensor
    state: torch.Tensor
    projection: OracleProjection
    relative_update: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True)
class AdaptiveResult:
    state: torch.Tensor
    steps: tuple[AdaptiveStep, ...]
    calls: torch.Tensor
    stop_reasons: tuple[str, ...]


def progress_to_rho_time(
    progress: torch.Tensor | float,
    *,
    epsilon: float = 0.002,
    terminal: float = 5.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """Map normalized trajectory progress to the CDEQ rho time grid."""
    if not 0 < epsilon < terminal:
        raise ValueError("expected 0 < epsilon < terminal")
    if rho <= 0:
        raise ValueError("rho must be positive")
    value = torch.as_tensor(progress)
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("progress must be finite and in [0, 1]")
    start = value.new_tensor(epsilon).pow(1 / rho)
    end = value.new_tensor(terminal).pow(1 / rho)
    return (start + value * (end - start)).pow(rho)


def rho_time_to_progress(
    time: torch.Tensor | float,
    *,
    epsilon: float = 0.002,
    terminal: float = 5.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """Inverse of :func:`progress_to_rho_time` on ``[epsilon, terminal]``."""
    if not 0 < epsilon < terminal:
        raise ValueError("expected 0 < epsilon < terminal")
    if rho <= 0:
        raise ValueError("rho must be positive")
    value = torch.as_tensor(time)
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all() or ((value < epsilon) | (value > terminal)).any():
        raise ValueError("time must be finite and in [epsilon, terminal]")
    start = value.new_tensor(epsilon).pow(1 / rho)
    end = value.new_tensor(terminal).pow(1 / rho)
    return ((value.pow(1 / rho) - start) / (end - start)).clamp(0, 1)


def _as_batched_projection_inputs(
    state: torch.Tensor,
    trajectory: torch.Tensor,
    state_mask: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    unbatched = state.ndim == 2
    if unbatched:
        state = state.unsqueeze(0)
    if trajectory.ndim == 3:
        trajectory = trajectory.unsqueeze(0)
    if state.ndim != 3 or trajectory.ndim != 4:
        raise ValueError(
            "state and trajectory must have shapes [B,L,H] and [B,K,L,H] "
            "(or their unbatched equivalents)"
        )
    batch, _, block, hidden = trajectory.shape
    if state.shape != (batch, block, hidden):
        raise ValueError("state and trajectory batch/block/hidden dimensions must match")

    state_mask = torch.as_tensor(state_mask, device=trajectory.device, dtype=torch.bool)
    if state_mask.ndim == 1:
        state_mask = state_mask.unsqueeze(0)
    if state_mask.shape != trajectory.shape[:2]:
        raise ValueError("state_mask must have shape [B,K] or [K]")

    token_mask = torch.as_tensor(token_mask, device=trajectory.device, dtype=torch.bool)
    if token_mask.ndim == 1:
        token_mask = token_mask.unsqueeze(0)
    if token_mask.ndim == 2:
        if token_mask.shape != (batch, block):
            raise ValueError("token_mask must have shape [B,L], [B,K,L], or [L]")
        token_mask = token_mask[:, None, :].expand(batch, trajectory.shape[1], block)
    elif token_mask.ndim == 3:
        if token_mask.shape != trajectory.shape[:3]:
            raise ValueError("per-state token_mask must have shape [B,K,L]")
    else:
        raise ValueError("token_mask must have shape [B,L], [B,K,L], or [L]")
    return state, trajectory, state_mask, token_mask, unbatched


def project_to_teacher_trajectory(
    state: torch.Tensor,
    trajectory: torch.Tensor,
    state_mask: torch.Tensor,
    token_mask: torch.Tensor,
) -> OracleProjection:
    """Project states onto all valid line segments of teacher trajectories.

    Projection coefficients minimize masked squared Euclidean error. Candidate
    segments are then compared using masked relative hidden error, making the
    score comparable across examples and token-mask lengths. ``margin`` is the
    second-best minus best normalized distance; a small value exposes an
    ambiguous oracle assignment.
    """
    state, trajectory, state_mask, token_mask, unbatched = _as_batched_projection_inputs(
        state, trajectory, state_mask, token_mask
    )
    if not torch.is_floating_point(state) or not torch.is_floating_point(trajectory):
        raise ValueError("state and trajectory must be floating point tensors")
    if trajectory.shape[1] < 2:
        raise ValueError("trajectory must provide space for at least two states")

    # A segment is usable only when both states and the corresponding tokens
    # are valid. Intersecting per-state token masks also handles EOS movement.
    segment_valid = state_mask[:, :-1] & state_mask[:, 1:]
    segment_token_mask = token_mask[:, :-1] & token_mask[:, 1:]
    segment_valid &= segment_token_mask.any(dim=-1)
    expanded_mask = segment_token_mask.unsqueeze(-1).to(dtype=trajectory.dtype)

    left = trajectory[:, :-1]
    delta = trajectory[:, 1:] - left
    difference = state[:, None] - left
    numerator = (difference * delta * expanded_mask).flatten(2).sum(dim=-1)
    denominator = (delta.square() * expanded_mask).flatten(2).sum(dim=-1)
    fraction = (numerator / denominator.clamp_min(torch.finfo(trajectory.dtype).eps)).clamp(0, 1)
    # A zero-length segment is still a meaningful point candidate.
    fraction = torch.where(denominator > 0, fraction, torch.zeros_like(fraction))
    candidate = left + fraction[:, :, None, None] * delta

    residual_norm = ((state[:, None] - candidate) * expanded_mask).flatten(2).norm(dim=-1)
    reference_norm = (candidate * expanded_mask).flatten(2).norm(dim=-1)
    scale = reference_norm.clamp_min(torch.finfo(trajectory.dtype).eps)
    distances = residual_norm / scale
    distances = distances.masked_fill(~segment_valid, float("inf"))

    best_distance, best_segment = distances.min(dim=1)
    valid = torch.isfinite(best_distance)
    if distances.shape[1] > 1:
        second_distance = distances.topk(2, dim=1, largest=False).values[:, 1]
    else:
        second_distance = torch.full_like(best_distance, float("inf"))
    margin = second_distance - best_distance

    batch_index = torch.arange(state.shape[0], device=state.device)
    safe_segment = best_segment.clamp_min(0)
    best_fraction = fraction[batch_index, safe_segment]
    best_candidate = candidate[batch_index, safe_segment]
    # Use ranks instead of raw padded indices, so progress remains correct for
    # any valid-state mask, including a non-prefix mask used in diagnostics.
    state_rank = state_mask.long().cumsum(dim=1) - 1
    segment_rank = state_rank[:, :-1][batch_index, safe_segment]
    state_count = state_mask.sum(dim=1)
    progress = (segment_rank + best_fraction) / (state_count - 1).clamp_min(1)

    invalid_index = torch.full_like(best_segment, -1)
    nan_progress = torch.full_like(progress, float("nan"))
    best_segment = torch.where(valid, best_segment, invalid_index)
    best_fraction = torch.where(valid, best_fraction, nan_progress)
    progress = torch.where(valid, progress.clamp(0, 1), nan_progress)
    best_candidate = torch.where(valid[:, None, None], best_candidate, state)

    def maybe_squeeze(value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(0) if unbatched else value

    return OracleProjection(
        progress=maybe_squeeze(progress),
        distance=maybe_squeeze(best_distance),
        second_distance=maybe_squeeze(second_distance),
        margin=maybe_squeeze(margin),
        segment_index=maybe_squeeze(best_segment),
        segment_fraction=maybe_squeeze(best_fraction),
        projected_state=maybe_squeeze(best_candidate),
        valid=maybe_squeeze(valid),
    )


def _relative_update(
    before: torch.Tensor, after: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    mask = token_mask.unsqueeze(-1).to(dtype=before.dtype)
    numerator = ((after - before) * mask).flatten(1).norm(dim=-1)
    denominator = (before * mask).flatten(1).norm(dim=-1)
    return numerator / denominator.clamp_min(torch.finfo(before.dtype).eps)


def _batched_projection(projection: OracleProjection) -> OracleProjection:
    """Restore the batch dimension for a projection made from a size-one batch."""
    if projection.progress.ndim > 0:
        return projection
    return OracleProjection(
        progress=projection.progress.unsqueeze(0),
        distance=projection.distance.unsqueeze(0),
        second_distance=projection.second_distance.unsqueeze(0),
        margin=projection.margin.unsqueeze(0),
        segment_index=projection.segment_index.unsqueeze(0),
        segment_fraction=projection.segment_fraction.unsqueeze(0),
        projected_state=projection.projected_state.unsqueeze(0),
        valid=projection.valid.unsqueeze(0),
    )


@torch.no_grad()
def adaptive_oracle_recurrence(
    adapter: AdaptiveAdapter,
    initial_state: torch.Tensor,
    trajectory: torch.Tensor,
    state_mask: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    max_calls: int = 4,
    epsilon: float = 0.002,
    terminal: float | None = None,
    rho: float = 7.0,
    identity_cap_progress: float = 0.995,
    regression_tolerance: float = 0.05,
    endpoint_progress: float = 0.999,
    endpoint_distance: float = 1e-3,
    max_oracle_distance: float | None = None,
    update_tolerance: float | None = None,
) -> AdaptiveResult:
    """Run the Stage-A adapter-only recurrence with continuous oracle times.

    The initializer is applied exactly once. The first updater call uses the
    existing ``t=0`` inference boundary; later calls use oracle progress. Until
    the endpoint is verified, progress is capped below one to avoid entering
    the exact ``t=T`` identity boundary. Stop decisions are made per example.
    """
    if max_calls not in (1, 2, 3, 4):
        raise ValueError("Stage A max_calls must be one of 1, 2, 3, or 4")
    if not 0 <= regression_tolerance <= 1:
        raise ValueError("regression_tolerance must be in [0, 1]")
    if not 0 <= identity_cap_progress < 1:
        raise ValueError("identity_cap_progress must be in [0, 1)")
    if not identity_cap_progress <= endpoint_progress <= 1:
        raise ValueError("endpoint_progress must be in [identity_cap_progress, 1]")
    if endpoint_distance < 0 or (max_oracle_distance is not None and max_oracle_distance < 0):
        raise ValueError("distance thresholds must be non-negative")
    if update_tolerance is not None and update_tolerance < 0:
        raise ValueError("update_tolerance must be non-negative")

    unbatched = initial_state.ndim == 2
    state = initial_state.unsqueeze(0) if unbatched else initial_state
    teacher = trajectory.unsqueeze(0) if trajectory.ndim == 3 else trajectory
    state_mask = torch.as_tensor(state_mask, device=state.device, dtype=torch.bool)
    token_mask = torch.as_tensor(token_mask, device=state.device, dtype=torch.bool)
    teacher_state_mask = state_mask.unsqueeze(0) if state_mask.ndim == 1 else state_mask
    inference_token_mask = token_mask
    if inference_token_mask.ndim == 1:
        inference_token_mask = inference_token_mask.unsqueeze(0)
    if inference_token_mask.ndim == 3:
        # Recurrence update norms use the conservative intersection across the
        # valid teacher states; the projector retains the full per-state mask.
        inference_token_mask = inference_token_mask.all(dim=1)
    if state.ndim != 3:
        raise ValueError("initial_state must have shape [B,L,H] or [L,H]")
    batch = state.shape[0]
    if inference_token_mask.shape != state.shape[:2]:
        raise ValueError("token_mask does not match initial_state")
    if not inference_token_mask.any(dim=1).all():
        raise ValueError("each example must have at least one valid token")

    resolved_terminal = float(adapter.terminal if terminal is None else terminal)
    if terminal is not None and abs(float(adapter.terminal) - resolved_terminal) > 1e-8:
        raise ValueError("terminal must match adapter.terminal")

    state = adapter.initialize(state)
    active = torch.ones(batch, device=state.device, dtype=torch.bool)
    calls = torch.zeros(batch, device=state.device, dtype=torch.long)
    reasons = ["" for _ in range(batch)]
    previous_progress = torch.full(
        (batch,), float("nan"), device=state.device, dtype=state.dtype
    )
    steps: list[AdaptiveStep] = []

    for call_index in range(1, max_calls + 1):
        active_before_call = active.clone()
        if call_index == 1:
            input_time = torch.zeros(batch, device=state.device, dtype=state.dtype)
        else:
            # Inactive examples receive the exact identity time while active
            # examples remain strictly below it.
            safe_progress = previous_progress.nan_to_num(0.0).clamp_max(identity_cap_progress)
            input_time = progress_to_rho_time(
                safe_progress,
                epsilon=epsilon,
                terminal=resolved_terminal,
                rho=rho,
            ).to(device=state.device, dtype=state.dtype)
            input_time = torch.where(
                active_before_call,
                input_time,
                torch.full_like(input_time, resolved_terminal),
            )

        before = state
        candidate = adapter.consistency(before, input_time)
        state = torch.where(active_before_call[:, None, None], candidate, before)
        calls += active_before_call.long()
        relative_update = _relative_update(before, state, inference_token_mask)
        projection = _batched_projection(
            project_to_teacher_trajectory(state, teacher, teacher_state_mask, token_mask)
        )
        steps.append(
            AdaptiveStep(
                call_index=call_index,
                input_time=input_time.clone(),
                state=state.clone(),
                projection=projection,
                relative_update=relative_update,
                active=active_before_call,
            )
        )

        invalid = active & ~projection.valid
        for index in invalid.nonzero(as_tuple=False).flatten().tolist():
            reasons[index] = "invalid_oracle"
        active &= projection.valid

        if max_oracle_distance is not None:
            off_manifold = active & (projection.distance > max_oracle_distance)
            for index in off_manifold.nonzero(as_tuple=False).flatten().tolist():
                reasons[index] = "off_manifold"
            active &= ~off_manifold

        has_previous = torch.isfinite(previous_progress)
        regressed = (
            active
            & has_previous
            & (projection.progress < previous_progress - regression_tolerance)
        )
        for index in regressed.nonzero(as_tuple=False).flatten().tolist():
            reasons[index] = "regression"
        active &= ~regressed

        endpoint = (
            active
            & (projection.progress >= endpoint_progress)
            & (projection.distance <= endpoint_distance)
        )
        for index in endpoint.nonzero(as_tuple=False).flatten().tolist():
            reasons[index] = "endpoint"
        active &= ~endpoint

        if update_tolerance is not None and call_index > 1:
            stalled = active & (relative_update <= update_tolerance)
            for index in stalled.nonzero(as_tuple=False).flatten().tolist():
                reasons[index] = "stalled"
            active &= ~stalled

        accepted = active & projection.valid
        conservative = projection.progress.clamp_max(identity_cap_progress)
        previous_progress = torch.where(
            accepted,
            torch.where(
                has_previous,
                torch.maximum(previous_progress, conservative),
                conservative,
            ),
            previous_progress,
        )
        if not active.any():
            break

    for index in active.nonzero(as_tuple=False).flatten().tolist():
        reasons[index] = "budget"
    # Defensive fallback for a future stop branch accidentally missing a label.
    reasons = [reason or "budget" for reason in reasons]
    result_state = state.squeeze(0) if unbatched else state
    result_calls = calls.squeeze(0) if unbatched else calls
    return AdaptiveResult(
        state=result_state,
        steps=tuple(steps),
        calls=result_calls,
        stop_reasons=tuple(reasons),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory-progress-aware oracle CDEQ recurrence"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--max-calls", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--time-mode", choices=("oracle",), default="oracle")
    parser.add_argument("--weights", choices=("ema", "online"), default="ema")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-file")
    parser.add_argument("--allow-split-mismatch", action="store_true")
    parser.add_argument("--max-oracle-distance", type=float)
    parser.add_argument("--endpoint-distance", type=float, default=1e-3)
    parser.add_argument("--regression-tolerance", type=float, default=1.0)
    parser.add_argument("--update-tolerance", type=float)
    return parser.parse_args()


def _state_metrics(
    state: torch.Tensor,
    endpoint: torch.Tensor,
    endpoint_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> dict[str, float | int]:
    mask = token_mask.unsqueeze(-1)
    relative = (
        ((state - endpoint) * mask).flatten(1).norm(dim=1)
        / (endpoint * mask).flatten(1).norm(dim=1).clamp_min(1e-8)
    )
    predicted_tokens = F.linear(state, lm_head_weight).argmax(dim=-1)
    correct = predicted_tokens.eq(endpoint_tokens) & token_mask
    return {
        "error_sum": float(relative.sum()),
        "correct_tokens": int(correct.sum()),
        "token_count": int(token_mask.sum()),
        "exact_blocks": int((correct | ~token_mask).all(dim=1).sum()),
        "examples": int(state.shape[0]),
    }


def _empty_call_totals() -> dict[str, float | int]:
    return {
        "error_sum": 0.0,
        "correct_tokens": 0,
        "token_count": 0,
        "exact_blocks": 0,
        "examples": 0,
        "projection_distance_sum": 0.0,
        "projection_progress_sum": 0.0,
        "projection_margin_sum": 0.0,
        "projection_margin_valid": 0,
        "projection_valid": 0,
        "relative_update_sum": 0.0,
        "active_calls": 0,
    }


def _finalize_call(total: dict[str, float | int], call_index: int) -> dict[str, float | int]:
    examples = max(int(total["examples"]), 1)
    token_count = max(int(total["token_count"]), 1)
    projection_valid = max(int(total["projection_valid"]), 1)
    projection_margin_valid = max(int(total["projection_margin_valid"]), 1)
    active_calls = max(int(total["active_calls"]), 1)
    return {
        "call": call_index,
        "endpoint_relative_error": float(total["error_sum"]) / examples,
        "endpoint_token_agreement": int(total["correct_tokens"]) / token_count,
        "exact_block_match": int(total["exact_blocks"]) / examples,
        "projection_valid_rate": int(total["projection_valid"]) / examples,
        "projection_distance": float(total["projection_distance_sum"]) / projection_valid,
        "projection_progress": float(total["projection_progress_sum"]) / projection_valid,
        "projection_margin": float(total["projection_margin_sum"])
        / projection_margin_valid,
        "relative_update": float(total["relative_update_sum"]) / active_calls,
        "active_call_fraction": int(total["active_calls"]) / examples,
        "examples": int(total["examples"]),
    }


def _gate_summary(per_call: list[dict[str, float | int]]) -> dict[str, object]:
    first = per_call[0]
    candidates = per_call[1:]
    if not candidates:
        return {
            "passed": False,
            "reason": "at least two calls are required for the recurrence gate",
        }
    best_error = min(candidates, key=lambda row: float(row["endpoint_relative_error"]))
    best_token = max(candidates, key=lambda row: float(row["endpoint_token_agreement"]))
    error_gain = (
        float(first["endpoint_relative_error"])
        - float(best_error["endpoint_relative_error"])
    ) / max(float(first["endpoint_relative_error"]), 1e-12)
    token_gain = float(best_token["endpoint_token_agreement"]) - float(
        first["endpoint_token_agreement"]
    )
    error_candidate_token_delta = float(best_error["endpoint_token_agreement"]) - float(
        first["endpoint_token_agreement"]
    )
    token_candidate_error_delta = (
        float(best_token["endpoint_relative_error"])
        - float(first["endpoint_relative_error"])
    ) / max(float(first["endpoint_relative_error"]), 1e-12)
    error_pass = error_gain >= 0.05 and error_candidate_token_delta >= -0.01
    token_pass = token_gain >= 0.02 and token_candidate_error_delta <= 0.01
    return {
        "passed": bool(error_pass or token_pass),
        "criteria": {
            "endpoint_error_relative_gain": 0.05,
            "token_agreement_gain": 0.02,
            "other_metric_max_regression": 0.01,
        },
        "best_error_call": int(best_error["call"]),
        "best_token_call": int(best_token["call"]),
        "endpoint_error_relative_gain": error_gain,
        "token_agreement_gain": token_gain,
        "best_error_call_token_delta": error_candidate_token_delta,
        "best_token_call_error_relative_delta": token_candidate_error_delta,
    }


@torch.inference_mode()
def run_oracle_cache_gate(args: argparse.Namespace) -> dict[str, object]:
    from safetensors.torch import load_file

    from .cache import HiddenTrajectoryDataset, iter_shard_batches
    from .config import load_config, resolve_repo_path
    from .runtime import move_batch, seed_everything
    from .train import build_adapter, load_checkpoint

    config = load_config(args.config)
    seed_everything(int(config["training"]["seed"]))
    device = torch.device(args.device)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if package["upstream"] != config["upstream"]:
        raise ValueError("checkpoint and adaptive config use different upstream revisions")
    checkpoint_config = package["config"]
    adapter = build_adapter(
        checkpoint_config,
        bool(package["use_initializer"]),
        rank=int(checkpoint_config["model"]["bottleneck_rank"]),
    )
    load_checkpoint(args.checkpoint, adapter)
    if args.weights == "ema":
        adapter.load_state_dict(package["ema_state"])
    adapter = adapter.to(device=device, dtype=torch.float32).eval()

    cache_root = resolve_repo_path(config, config["paths"]["cache_dir"])
    dataset = HiddenTrajectoryDataset(cache_root / args.split / "manifest.json")
    split_matches = dataset.manifest["data_split_hash"] == package["data_split_hash"]
    if not split_matches and not args.allow_split_mismatch:
        raise ValueError(
            "checkpoint/cache split hash mismatch; use --allow-split-mismatch only for "
            "an explicitly labelled diagnostic smoke run"
        )
    lm_head = load_file(str(cache_root / "lm_head.safetensors"), device=str(device))["weight"]
    lm_head = lm_head.float()

    batch_size = args.batch_size or int(config["training"]["batch_size"])
    limit = args.sample_limit or len(dataset)
    totals = [_empty_call_totals() for _ in range(args.max_calls)]
    stop_reasons: Counter[str] = Counter()
    all_calls: list[int] = []
    examples = 0
    for batch in iter_shard_batches(dataset, batch_size, shuffle=False):
        if examples >= limit:
            break
        take = min(next(iter(batch.values())).shape[0], limit - examples)
        batch = {name: value[:take] for name, value in batch.items()}
        batch = move_batch(batch, device, floating_dtype=torch.float32)
        states = batch["states"]
        state_mask = batch["state_mask"]
        endpoint = states[
            torch.arange(states.shape[0], device=device), state_mask.sum(dim=1) - 1
        ]
        result = adaptive_oracle_recurrence(
            adapter,
            states[:, 0],
            states,
            state_mask,
            batch["token_mask"],
            max_calls=args.max_calls,
            epsilon=float(config["time"]["epsilon"]),
            terminal=float(config["time"]["terminal"]),
            rho=float(config["time"]["rho"]),
            endpoint_distance=args.endpoint_distance,
            regression_tolerance=args.regression_tolerance,
            max_oracle_distance=args.max_oracle_distance,
            update_tolerance=args.update_tolerance,
        )
        last_state = states[:, 0]
        for call_index in range(args.max_calls):
            if call_index < len(result.steps):
                step = result.steps[call_index]
                last_state = step.state
                projection = step.projection
                valid = projection.valid
                active = step.active
                total = totals[call_index]
                total["projection_distance_sum"] += float(projection.distance[valid].sum())
                total["projection_progress_sum"] += float(projection.progress[valid].sum())
                finite_margin = valid & torch.isfinite(projection.margin)
                total["projection_margin_sum"] += float(projection.margin[finite_margin].sum())
                total["projection_margin_valid"] += int(finite_margin.sum())
                total["projection_valid"] += int(valid.sum())
                total["relative_update_sum"] += float(step.relative_update[active].sum())
                total["active_calls"] += int(active.sum())
            metrics = _state_metrics(
                last_state,
                endpoint,
                batch["endpoint_tokens"],
                batch["token_mask"],
                lm_head,
            )
            for key, value in metrics.items():
                totals[call_index][key] += value
        stop_reasons.update(result.stop_reasons)
        all_calls.extend(int(value) for value in result.calls.detach().cpu().flatten())
        examples += take

    per_call = [_finalize_call(total, index + 1) for index, total in enumerate(totals)]
    calls = torch.tensor(all_calls, dtype=torch.float32)
    report: dict[str, object] = {
        "schema_version": "llm_cdeq_adaptive_oracle_report_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": args.weights,
        "use_initializer": bool(package["use_initializer"]),
        "time_mode": args.time_mode,
        "split": args.split,
        "sample_limit": limit,
        "examples": examples,
        "max_calls": args.max_calls,
        "checkpoint_split_hash": package["data_split_hash"],
        "cache_split_hash": dataset.manifest["data_split_hash"],
        "split_hash_matches": split_matches,
        "diagnostic_split_mismatch": not split_matches,
        "per_call": per_call,
        "gate": _gate_summary(per_call),
        "calls": {
            "mean": float(calls.mean()) if calls.numel() else 0.0,
            "median": float(calls.median()) if calls.numel() else 0.0,
            "p95": float(torch.quantile(calls, 0.95)) if calls.numel() else 0.0,
        },
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "thresholds": {
            "endpoint_distance": args.endpoint_distance,
            "max_oracle_distance": args.max_oracle_distance,
            "regression_tolerance": args.regression_tolerance,
            "update_tolerance": args.update_tolerance,
        },
    }
    return report


def main() -> None:
    args = _parse_args()
    report = run_oracle_cache_gate(args)
    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
