import random
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_cdeq.cllm_step import OfficialCLLMSingleStep
from llm_cdeq.corrector import TransformerResidualCorrector
from llm_cdeq import evaluate_wrapped
from llm_cdeq.evaluate_wrapped import (
    aggregate_diagnostics,
    generate_answer,
    gsm8k_phase_decision,
    initial_block,
    refine_block,
)
from llm_cdeq.wrapped_model import WrappedCLLM


def make_wrapper():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    operator = OfficialCLLMSingleStep(
        model, block_size=4, model_source="tiny-test", enforce_official=False
    )
    corrector = TransformerResidualCorrector(
        hidden_size=16,
        rank=8,
        block_size=4,
        layers=1,
        heads=2,
        ffn_size=16,
        terminal=5.0,
    )
    return WrappedCLLM(operator, corrector)


def inputs(wrapper):
    prompt = torch.tensor([[1, 4, 7]])
    prefill = wrapper.prefill(prompt)
    current = torch.cat((prefill.first_token, torch.tensor([[5, 6, 7]])), dim=1)
    return prefill, current


def test_initial_block_preserves_first_token_and_is_deterministic():
    first = torch.tensor([[7]])
    prefix = torch.tensor([[1, 2, 3, 4]])
    left = initial_block(first, prefix, 4, random.Random(42))
    right = initial_block(first, prefix, 4, random.Random(42))
    assert torch.equal(left, right)
    assert left.shape == (1, 4)
    assert int(left[0, 0]) == 7


def test_refine_block_counts_official_and_wrapped_calls():
    wrapper = make_wrapper()
    prefill, current = inputs(wrapper)
    _, official = refine_block(
        wrapper, prefill, current, method="official_single", max_rounds=4
    )
    _, wrapped = refine_block(
        wrapper, prefill, current, method="wrapped", max_rounds=4
    )
    assert official["rounds"] == official["cllm_backbone_nfe"] == 1
    assert official["corrector_nfe"] == 0
    assert wrapped["rounds"] == wrapped["cllm_backbone_nfe"] == 1
    assert wrapped["corrector_nfe"] == 1


def test_official_fixed_reports_nonconvergence_without_aborting(monkeypatch):
    class Tokenizer:
        eos_token_id = 2

        def __call__(self, _prompt, return_tensors):
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.tensor([[1, 3]]))

    wrapper = SimpleNamespace(
        cllm_step=SimpleNamespace(model=torch.nn.Linear(1, 1)),
        prefill=lambda _prefix: SimpleNamespace(
            first_token=torch.tensor([[7]]), prompt_nfe=1
        ),
    )
    monkeypatch.setattr(
        evaluate_wrapped,
        "refine_block",
        lambda *_args, **_kwargs: (
            torch.tensor([[7]]),
            {
                "rounds": 17,
                "cllm_backbone_nfe": 17,
                "corrector_nfe": 0,
                "initializer_nfe": 0,
                "corrector_seconds": 0.0,
                "converged": False,
            },
        ),
    )
    _, metrics = generate_answer(
        wrapper,
        Tokenizer(),
        "prompt",
        method="official_fixed",
        block_size=1,
        max_new_tokens=1,
        max_rounds=17,
        seed=42,
    )
    assert metrics["converged_blocks"] == 0
    assert metrics["all_blocks_converged"] is False


def test_generation_detects_constant_and_adjacent_duplicate_blocks(monkeypatch):
    class Tokenizer:
        eos_token_id = 2

        def __call__(self, _prompt, return_tensors):
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.tensor([[1, 3]]))

    wrapper = SimpleNamespace(
        cllm_step=SimpleNamespace(model=torch.nn.Linear(1, 1)),
        prefill=lambda _prefix: SimpleNamespace(
            first_token=torch.tensor([[7]]), prompt_nfe=1
        ),
    )
    block_metrics = {
        "rounds": 1,
        "cllm_backbone_nfe": 1,
        "corrector_nfe": 1,
        "initializer_nfe": 0,
        "corrector_seconds": 0.0,
        "converged": True,
    }

    outputs = iter((torch.tensor([[4, 5]]), torch.tensor([[4, 5]])))
    monkeypatch.setattr(
        evaluate_wrapped,
        "refine_block",
        lambda *_args, **_kwargs: (next(outputs), block_metrics),
    )
    _, duplicate = generate_answer(
        wrapper,
        Tokenizer(),
        "prompt",
        method="wrapped",
        block_size=2,
        max_new_tokens=4,
        max_rounds=1,
        seed=42,
    )
    assert duplicate["constant_token_block"] is False
    assert duplicate["adjacent_duplicate_block"] is True
    assert duplicate["repeated_block"] is True
    assert duplicate["prompt_prefill_nfe"] == 2

    monkeypatch.setattr(
        evaluate_wrapped,
        "refine_block",
        lambda *_args, **_kwargs: (torch.tensor([[7, 7]]), block_metrics),
    )
    _, constant = generate_answer(
        wrapper,
        Tokenizer(),
        "prompt",
        method="wrapped",
        block_size=2,
        max_new_tokens=2,
        max_rounds=1,
        seed=42,
    )
    assert constant["constant_token_block"] is True
    assert constant["adjacent_duplicate_block"] is False
    assert constant["repeated_block"] is True


def test_aggregate_diagnostics_includes_health_convergence_and_prefill():
    row = {
        "empty_decoded": False,
        "eos_first_token": False,
        "eos_reached": True,
        "truncated": False,
        "all_blocks_converged": True,
        "constant_token_block": False,
        "adjacent_duplicate_block": True,
        "repeated_block": True,
        "generated_tokens": 17,
        "prompt_prefill_nfe": 3,
        "cllm_backbone_nfe": 9,
        "corrector_nfe": 3,
        "initializer_nfe": 0,
        "wall_seconds": 0.25,
    }
    totals = aggregate_diagnostics([row, row])
    assert totals["empty_outputs"] == 0
    assert totals["eos_reached"] == 2
    assert totals["truncated"] == 0
    assert totals["all_blocks_converged"] == 2
    assert totals["prompt_prefill_nfe"] == 6
    assert totals["adjacent_duplicate_block"] == totals["repeated_block"] == 2


def test_phase_decision_uses_precommitted_weights_and_config_thresholds():
    row = {
        "empty_decoded": False,
        "eos_first_token": False,
        "eos_reached": True,
        "truncated": False,
        "all_blocks_converged": True,
        "constant_token_block": False,
        "adjacent_duplicate_block": False,
        "repeated_block": False,
        "generated_tokens": 17,
        "prompt_prefill_nfe": 1,
        "cllm_backbone_nfe": 1,
        "corrector_nfe": 1,
        "initializer_nfe": 0,
        "wall_seconds": 0.25,
    }
    healthy = aggregate_diagnostics([row] * 8)
    healthy["cllm_backbone_nfe"] = 963
    reports = {
        "official_fixed": {**healthy, "score": {"correct": 6}},
        "wrapped_online": {**healthy, "score": {"correct": 8}},
        "wrapped_ema": {**healthy, "score": {"correct": 4}},
    }
    decision = gsm8k_phase_decision(
        reports, selected_weights="ema", evaluation={}, sample_limit=8
    )
    assert decision["selected_method"] == "wrapped_ema"
    assert decision["engineering_control_passed"] is True
    assert decision["selected_health_passed"] is True
    assert decision["base_signal"] is True
    assert decision["next_phase_allowed"] is True

    stricter = gsm8k_phase_decision(
        reports,
        selected_weights="ema",
        evaluation={"gsm8k_phase_correct_min": 5},
        sample_limit=8,
    )
    assert stricter["base_signal"] is True
    assert stricter["next_phase_allowed"] is False

    cache_failed = gsm8k_phase_decision(
        reports,
        selected_weights="ema",
        evaluation={},
        sample_limit=8,
        cache_gate_passed=False,
    )
    assert cache_failed["base_signal"] is True
    assert cache_failed["next_phase_allowed"] is False

    reports["official_fixed"]["cllm_backbone_nfe"] = 962
    nfe_drift = gsm8k_phase_decision(
        reports, selected_weights="ema", evaluation={}, sample_limit=8
    )
    assert nfe_drift["engineering_control_passed"] is False
