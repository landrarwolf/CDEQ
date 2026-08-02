import copy

import pytest
import torch

from llm_cdeq.config import (
    CACHE_TIME_GRID_CONTRACT,
    cache_config_digest,
    config_digest,
)
from llm_cdeq.corrector import TransformerResidualCorrector
from llm_cdeq.train_wrapped import (
    WrappedMetrics,
    _trainable_token_mask,
    gate_results,
    load_wrapped_checkpoint,
    prepare_output_dir,
    save_wrapped_checkpoint,
    select_validation_weights,
    selection_key,
    validate_cache_pair,
    validate_resume_history,
    wrapped_train_step,
)


def config():
    return {
        "schema_version": "llm_cdeq_config_v1",
        "upstream": {"cllm_model_id": "cllm/consistency-llm-7b-math"},
        "model": {
            "operator": "official_cllm",
            "hidden_size": 8,
            "block_size": 2,
            "corrector_rank": 4,
            "corrector_layers": 1,
            "corrector_heads": 2,
            "corrector_ffn_size": 8,
            "cache_schema": "llm_cdeq_official_cllm_hidden_cache_v1",
            "max_trajectory_states": 17,
        },
        "time": {"epsilon": 0.002, "terminal": 5.0, "rho": 7.0},
        "training": {
            "seed": 42,
            "train_limit": 512,
            "validation_limit": 128,
            "shard_size": 64,
            "local_weight": 0.1,
            "endpoint_weight": 0.9,
            "safe_margin": 0.0,
            "safe_weight": 0.1,
            "token_ce_weight": 0.05,
            "checkpoint_selection": "online_endpoint_error",
        },
        "evaluation": {
            "attention_backend": "sdpa",
            "endpoint_error_improvement_gate": 0.2,
            "token_agreement_drop_gate": 0.01,
            "safe_violation_rate_gate": 0.05,
        },
    }


def batch():
    start = torch.randn(4, 2, 8)
    endpoint = torch.randn(4, 2, 8)
    middle = 0.5 * (start + endpoint)
    return {
        "canonical_hidden": torch.stack((start, middle, endpoint), dim=1),
        "state_mask": torch.ones(4, 3, dtype=torch.bool),
        "time_grid": torch.tensor([[0.002, 1.0, 5.0]]).expand(4, -1).clone(),
        "endpoint_hidden": endpoint,
        "endpoint_tokens": torch.zeros(4, 2, dtype=torch.long),
        "eos_mask": torch.ones(4, 2, dtype=torch.bool),
    }


def test_wrapped_loss_is_finite_and_backpropagates_only_to_corrector():
    corrector = TransformerResidualCorrector(
        hidden_size=8,
        rank=4,
        block_size=2,
        layers=1,
        heads=2,
        ffn_size=8,
    )
    ema = copy.deepcopy(corrector)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    head = torch.randn(11, 8)
    loss, parts = wrapped_train_step(
        corrector,
        ema,
        batch(),
        head,
        config(),
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "main",
        "local",
        "endpoint",
        "safe",
        "safe_violation_rate",
        "token_ce",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in corrector.parameters())
    assert all(parameter.grad is None for parameter in ema.parameters())


def test_local_teacher_uses_later_boundary_and_deployment_branch_uses_state_zero():
    class RecordingCorrector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))
            self.calls = []

        def forward(self, state, time):
            self.calls.append((state.detach().clone(), time.detach().clone()))
            value = torch.as_tensor(time, dtype=state.dtype, device=state.device)
            return state + self.scale * (1 - value[:, None, None] / 5.0)

    early = torch.zeros(1, 2, 8)
    endpoint = torch.ones(1, 2, 8)
    sample = {
        "canonical_hidden": torch.stack((early, endpoint), dim=1),
        "state_mask": torch.ones(1, 2, dtype=torch.bool),
        "time_grid": torch.tensor([[0.002, 5.0]]),
        "endpoint_hidden": endpoint,
        "endpoint_tokens": torch.zeros(1, 2, dtype=torch.long),
        "eos_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    student = RecordingCorrector()
    teacher = RecordingCorrector()
    teacher.load_state_dict(student.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    cfg = config()
    cfg["training"].update(
        local_weight=1.0,
        endpoint_weight=0.0,
        safe_weight=0.0,
        token_ce_weight=0.0,
    )
    loss, _ = wrapped_train_step(
        student,
        teacher,
        sample,
        torch.randn(3, 8),
        cfg,
        generator=torch.Generator().manual_seed(0),
    )
    loss.backward()
    assert len(student.calls) == 2
    torch.testing.assert_close(student.calls[0][0], early)
    torch.testing.assert_close(student.calls[0][1], torch.zeros(1))
    torch.testing.assert_close(student.calls[1][0], early)
    torch.testing.assert_close(student.calls[1][1], torch.tensor([0.002]))
    assert len(teacher.calls) == 1
    torch.testing.assert_close(teacher.calls[0][0], endpoint)
    torch.testing.assert_close(teacher.calls[0][1], torch.tensor([5.0]))
    assert student.scale.grad is not None and float(student.scale.grad.abs()) > 0


def test_token_training_mask_excludes_anchor_and_post_eos_tail():
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    assert torch.equal(
        _trainable_token_mask(mask),
        torch.tensor([[False, True, True, False], [False, False, False, False]]),
    )


def test_wrapped_checkpoint_round_trip_contains_full_metadata(tmp_path):
    corrector = TransformerResidualCorrector(
        hidden_size=8, rank=4, block_size=2, layers=1, heads=2, ffn_size=8
    )
    ema = copy.deepcopy(corrector)
    optimizer = torch.optim.AdamW(corrector.parameters())
    metrics = WrappedMetrics(0.5, 1.0, 0.5, 0.7, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 4)
    manifest = {
        "backbone_parameter_count": 10_000,
        "backbone_checksum": "checksum",
        "data_split_hash": "split",
        "schema_version": "llm_cdeq_official_cllm_hidden_cache_v1",
        "operator": "official_cllm",
    }
    path = tmp_path / "wrapped.pt"
    save_wrapped_checkpoint(
        path,
        corrector,
        ema,
        optimizer,
        config(),
        manifest,
        metrics,
        global_step=7,
        epoch=2,
        best_endpoint_relative_error=0.5,
        ema_metrics=metrics,
        selected_weights="ema",
        current_selection_key=selection_key(metrics),
        best_selected_weights="ema",
        best_selection_key=selection_key(metrics),
        best_selected_epoch=2,
    )
    restored = copy.deepcopy(corrector)
    package = load_wrapped_checkpoint(path, restored)
    assert package["schema_version"] == "llm_cdeq_wrapped_checkpoint_v1"
    assert package["operator"] == "official_cllm"
    assert package["epoch"] == 2
    assert package["epoch_complete"] is True
    assert package["validation_metrics"]["endpoint_relative_error"] == 0.5
    assert package["ema_validation_metrics"]["endpoint_relative_error"] == 0.5
    assert package["selected_weights"] == "ema"
    assert package["selected_epoch"] == 2
    assert package["best_selected_weights"] == "ema"
    assert package["best_selected_epoch"] == 2
    assert package["selection_key"] == list(selection_key(metrics))
    assert package["backbone_checksum_before"] == package["backbone_checksum_after"]
    for expected, actual in zip(corrector.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_gate_requires_all_fixed_acceptance_conditions():
    metrics = WrappedMetrics(0.7, 1.0, 0.3, 0.59, 0.6, 0.1, 0.1, 0.05, 0.0, 0.0, 64)
    assert all(gate_results(metrics, config()).values())


def test_token_aware_selection_prefers_token_then_exact_and_can_choose_ema():
    cfg = config()
    cfg["training"]["checkpoint_selection"] = "token_exact_any"
    online = WrappedMetrics(0.5, 1.0, 0.5, 0.72, 0.70, 0.20, 0.01, 0.0, 0.0, 0.0, 8)
    ema = WrappedMetrics(0.4, 1.0, 0.6, 0.73, 0.70, 0.10, 0.01, 0.0, 0.0, 0.0, 8)
    name, selected, key = select_validation_weights(online, ema, cfg)
    assert name == "ema"
    assert selected == ema
    assert key == selection_key(ema)

    equal_token = WrappedMetrics(
        0.4, 1.0, 0.6, 0.72, 0.70, 0.30, 0.01, 0.0, 0.0, 0.0, 8
    )
    name, selected, _ = select_validation_weights(online, equal_token, cfg)
    assert name == "ema"
    assert selected.exact_block_match > online.exact_block_match


def test_token_aware_selection_rejects_unsafe_candidate():
    cfg = config()
    cfg["training"]["checkpoint_selection"] = "token_exact_any"
    safe = WrappedMetrics(0.5, 1.0, 0.5, 0.72, 0.70, 0.20, 0.01, 0.0, 0.0, 0.0, 8)
    unsafe = WrappedMetrics(0.4, 1.0, 0.6, 0.99, 0.70, 0.90, 0.01, 0.1, 0.0, 0.0, 8)
    name, selected, _ = select_validation_weights(unsafe, safe, cfg)
    assert name == "ema"
    assert selected == safe


def test_output_directory_refuses_accidental_overwrite(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "history.jsonl").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_output_dir(output, resume=False)
    prepare_output_dir(output, resume=True)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_output_dir(tmp_path / "missing", resume=True)


def test_train_and_validation_cache_contract_must_match():
    cfg = config()
    common = {
        "schema_version": "llm_cdeq_official_cllm_hidden_cache_v1",
        "operator": "official_cllm",
        "data_split_hash": "split",
        "backbone_checksum": "backbone",
        "cllm_model_id": "cllm/consistency-llm-7b-math",
        "cllm_model_revision": "revision",
        "config_digest": config_digest(cfg),
        "time_grid_contract": CACHE_TIME_GRID_CONTRACT,
        "cache_config_digest": cache_config_digest(cfg),
        "data_id_overlap": 0,
    }

    class Dataset:
        def __init__(self, manifest):
            self.manifest = manifest

    train = Dataset({**common, "split": "train"})
    validation = Dataset({**common, "split": "validation"})
    validate_cache_pair(train, validation, cfg)
    tuning = copy.deepcopy(cfg)
    tuning["training"]["token_ce_weight"] = 0.2
    validate_cache_pair(train, validation, tuning)

    cache_change = copy.deepcopy(cfg)
    cache_change["time"]["terminal"] = 6.0
    with pytest.raises(ValueError, match="cache recipe"):
        validate_cache_pair(train, validation, cache_change)

    validation.manifest["data_split_hash"] = "other"
    with pytest.raises(ValueError, match="data_split_hash"):
        validate_cache_pair(train, validation, cfg)


def test_legacy_cache_keeps_full_config_digest_validation():
    cfg = config()
    common = {
        "schema_version": "llm_cdeq_official_cllm_hidden_cache_v1",
        "operator": "official_cllm",
        "data_split_hash": "split",
        "backbone_checksum": "backbone",
        "cllm_model_id": "cllm/consistency-llm-7b-math",
        "cllm_model_revision": "revision",
        "config_digest": config_digest(cfg),
        "data_id_overlap": 0,
    }

    class Dataset:
        def __init__(self, manifest):
            self.manifest = manifest

    train = Dataset({**common, "split": "train"})
    validation = Dataset({**common, "split": "validation"})
    validate_cache_pair(train, validation, cfg)
    changed = copy.deepcopy(cfg)
    changed["training"]["token_ce_weight"] = 0.2
    with pytest.raises(ValueError, match="legacy cache"):
        validate_cache_pair(train, validation, changed)


def test_resume_checkpoint_must_match_history_tail(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text('{"epoch":2,"global_step":24}\n', encoding="utf-8")
    validate_resume_history(history, {"epoch": 2, "global_step": 24})
    with pytest.raises(ValueError, match="history tail"):
        validate_resume_history(history, {"epoch": 1, "global_step": 16})
