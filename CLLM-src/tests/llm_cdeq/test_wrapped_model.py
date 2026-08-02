import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from llm_cdeq.cllm_step import OfficialCLLMSingleStep
from llm_cdeq.corrector import TransformerResidualCorrector
from llm_cdeq.wrapped_model import WrappedCLLM


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
    return OfficialCLLMSingleStep(
        LlamaForCausalLM(config).eval(),
        block_size=block_size,
        model_source="tiny-test",
        enforce_official=False,
    )


class CountingInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state):
        self.calls += 1
        return state + 0.01


def make_wrapper(initializer=None):
    operator = tiny_operator()
    corrector = TransformerResidualCorrector(
        hidden_size=16,
        rank=8,
        block_size=4,
        layers=1,
        heads=2,
        ffn_size=16,
        terminal=5.0,
    )
    return WrappedCLLM(operator, corrector, initializer=initializer)


def inputs(wrapper):
    prompt = torch.tensor([[1, 4, 7]])
    prefill = wrapper.prefill(prompt)
    current = torch.cat((prefill.first_token, torch.tensor([[5, 6, 7]])), dim=1)
    return prefill, current


def assert_matches_official(output):
    assert torch.equal(output.hidden, output.base.canonical_hidden)
    assert torch.equal(output.logits, output.base.logits)
    assert torch.equal(output.tokens, output.base.tokens)


def test_zero_residual_matches_official_cllm_exactly():
    wrapper = make_wrapper()
    prefill, current = inputs(wrapper)
    assert_matches_official(wrapper(prefill, current, torch.tensor(0.0), round_index=0))


def test_terminal_and_disable_adapter_match_official_cllm_exactly():
    initializer = CountingInitializer()
    wrapper = make_wrapper(initializer=initializer)
    with torch.no_grad():
        torch.nn.init.normal_(wrapper.corrector.up.weight)
    prefill, current = inputs(wrapper)
    terminal = wrapper(prefill, current, torch.tensor(5.0), round_index=0)
    disabled = wrapper(
        prefill,
        current,
        torch.tensor(0.0),
        round_index=0,
        disable_adapter=True,
    )
    assert_matches_official(terminal)
    assert_matches_official(disabled)
    assert initializer.calls == 0


def test_mixed_terminal_batch_preserves_official_hidden_for_terminal_sample():
    initializer = CountingInitializer()
    wrapper = make_wrapper(initializer=initializer)
    prompt = torch.tensor([[1, 4, 7], [1, 4, 7]])
    prefill = wrapper.prefill(prompt)
    suffix = torch.tensor([[5, 6, 7], [8, 9, 10]])
    current = torch.cat((prefill.first_token, suffix), dim=1)
    output = wrapper(
        prefill,
        current,
        torch.tensor([5.0, 0.0]),
        round_index=0,
    )
    assert torch.equal(output.hidden[0], output.base.canonical_hidden[0])
    assert initializer.calls == 1


def test_initializer_runs_only_on_first_round_and_every_round_reruns_backbone():
    initializer = CountingInitializer()
    wrapper = make_wrapper(initializer=initializer)
    prefill, current = inputs(wrapper)
    total_backbone_nfe = 0
    total_corrector_nfe = 0
    total_initializer_nfe = 0
    for round_index in range(3):
        output = wrapper(
            prefill, current, torch.tensor(0.0), round_index=round_index
        )
        current = output.tokens
        total_backbone_nfe += output.cllm_backbone_nfe
        total_corrector_nfe += output.corrector_nfe
        total_initializer_nfe += output.initializer_nfe
    assert initializer.calls == 1
    assert total_backbone_nfe == 3
    assert total_corrector_nfe == 3
    assert total_initializer_nfe == 1
    assert not any(parameter.requires_grad for parameter in wrapper.cllm_step.parameters())


def test_wrapper_bridges_bfloat16_backbone_and_float32_corrector():
    wrapper = make_wrapper()
    wrapper.cllm_step.model.to(torch.bfloat16)
    with torch.no_grad():
        torch.nn.init.normal_(wrapper.corrector.up.weight, std=1e-3)
    prefill, current = inputs(wrapper)
    output = wrapper(prefill, current, torch.tensor(0.0), round_index=0)
    assert output.base.canonical_hidden.dtype == torch.bfloat16
    assert output.hidden.dtype == torch.float32
    assert output.logits.dtype == torch.float32
    assert output.tokens.dtype == torch.long
