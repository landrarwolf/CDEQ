import copy

import torch

from llm_cdeq.corrector import TransformerResidualCorrector
from llm_cdeq.train_wrapped import (
    WrappedMetrics,
    gate_results,
    load_wrapped_checkpoint,
    save_wrapped_checkpoint,
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
        },
        "time": {"terminal": 5.0},
        "training": {
            "local_weight": 0.1,
            "endpoint_weight": 0.9,
            "safe_margin": 0.0,
            "safe_weight": 0.1,
            "token_ce_weight": 0.05,
        },
        "evaluation": {
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
    )
    restored = copy.deepcopy(corrector)
    package = load_wrapped_checkpoint(path, restored)
    assert package["schema_version"] == "llm_cdeq_wrapped_checkpoint_v1"
    assert package["operator"] == "official_cllm"
    assert package["backbone_checksum_before"] == package["backbone_checksum_after"]
    for expected, actual in zip(corrector.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_gate_requires_all_fixed_acceptance_conditions():
    metrics = WrappedMetrics(0.7, 1.0, 0.3, 0.59, 0.6, 0.1, 0.1, 0.05, 0.0, 0.0, 64)
    assert all(gate_results(metrics, config()).values())
