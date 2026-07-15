import torch

from llm_cdeq.corrector import (
    TransformerResidualCorrector,
    build_initializer,
    corrector_parameter_count,
)


def small_corrector():
    return TransformerResidualCorrector(
        hidden_size=16,
        rank=8,
        block_size=4,
        layers=1,
        heads=2,
        ffn_size=16,
        terminal=5.0,
    )


def test_zero_initialized_corrector_and_terminal_boundary_are_exact_identity():
    corrector = small_corrector()
    state = torch.randn(2, 4, 16)
    assert torch.equal(corrector(state, torch.zeros(2)), state)
    with torch.no_grad():
        torch.nn.init.normal_(corrector.up.weight)
    assert torch.equal(corrector(state, torch.full((2,), 5.0)), state)
    assert torch.equal(corrector(state, torch.zeros(2), disable_adapter=True), state)


def test_causal_attention_does_not_leak_future_positions():
    corrector = small_corrector().eval()
    with torch.no_grad():
        torch.nn.init.normal_(corrector.up.weight, std=0.02)
    left = torch.randn(1, 4, 16)
    right = left.clone()
    right[:, -1] += 10
    left_output = corrector(left, torch.zeros(1))
    right_output = corrector(right, torch.zeros(1))
    torch.testing.assert_close(left_output[:, :-1], right_output[:, :-1])


def test_default_corrector_and_initializer_stay_below_one_percent_of_backbone():
    corrector = TransformerResidualCorrector()
    initializer = build_initializer(4096, 512)
    total = corrector_parameter_count(corrector) + corrector_parameter_count(initializer)
    assert 7_000_000 < corrector_parameter_count(corrector) < 8_500_000
    assert total / 6_738_677_760 < 0.01
