import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_cdeq.cllm_step import OfficialCLLMSingleStep


def tiny_operator(block_size=4):
    config = LlamaConfig(
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
    model = LlamaForCausalLM(config).eval()
    return OfficialCLLMSingleStep(
        model, block_size=block_size, model_source="tiny-test", enforce_official=False
    )


def test_canonical_hidden_matches_official_jacobi_shift_and_prompt_cache_is_immutable():
    operator = tiny_operator()
    prompt = torch.tensor([[1, 4, 7]])
    prefill = operator.prefill(prompt)
    current = torch.cat((prefill.first_token, torch.tensor([[5, 6, 7]])), dim=1)
    first = operator(prefill, current)
    second = operator(prefill, first.tokens)
    assert torch.equal(first.logits.argmax(dim=-1), first.tokens)
    assert first.prompt_cache_lengths_before == first.prompt_cache_lengths_after
    assert second.prompt_cache_lengths_before == prefill.prompt_cache_lengths
    assert all(length == prompt.shape[1] for length in prefill.prompt_cache_lengths)
    assert first.backbone_nfe == second.backbone_nfe == 1


def test_official_operator_rejects_abel_or_unknown_model_source():
    config = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    model = LlamaForCausalLM(config)
    with pytest.raises(ValueError, match="consistency-llm-7b-math"):
        OfficialCLLMSingleStep(model, model_source="GAIR/Abel-7B-001")
