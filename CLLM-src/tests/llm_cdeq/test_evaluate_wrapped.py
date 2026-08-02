import random
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_cdeq.cllm_step import OfficialCLLMSingleStep
from llm_cdeq.corrector import TransformerResidualCorrector
from llm_cdeq import evaluate_wrapped
from llm_cdeq.evaluate_wrapped import generate_answer, initial_block, refine_block
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
