import torch

from llm_cdeq.model import (
    CDEQAdapter,
    RMSNormNoAffine,
    make_ema,
    trainable_parameter_count,
    update_ema_,
)


def test_rms_norm_matches_definition_without_parameters():
    norm = RMSNormNoAffine(4, eps=1e-6)
    value = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    expected = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(value), expected)
    assert not list(norm.parameters())


def test_terminal_boundary_is_exact_identity():
    adapter = CDEQAdapter(hidden_size=16, rank=4, terminal=5.0)
    state = torch.randn(3, 7, 16)
    result = adapter.consistency(state, torch.full((3,), 5.0))
    assert torch.equal(result, state)


def test_initializer_is_detached_from_consistency_update():
    adapter = CDEQAdapter(hidden_size=16, rank=4, use_initializer=True)
    state = torch.randn(2, 3, 16)
    output = adapter(
        state,
        torch.zeros(2),
        apply_initializer=True,
        detach_initializer=True,
    )
    output.square().mean().backward()
    assert all(parameter.grad is None for parameter in adapter.initializer_parameters())
    assert any(parameter.grad is not None for parameter in adapter.consistency_parameters())


def test_ema_updates_parameters_and_copies_integer_buffers():
    adapter = CDEQAdapter(hidden_size=8, rank=2)
    ema = make_ema(adapter)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.add_(1)
    before = next(ema.parameters()).clone()
    online = next(adapter.parameters()).clone()
    update_ema_(ema, adapter, 0.5)
    torch.testing.assert_close(next(ema.parameters()), 0.5 * before + 0.5 * online)
    assert not any(parameter.requires_grad for parameter in ema.parameters())


def test_adapter_is_below_one_percent_of_7b_backbone():
    adapter = CDEQAdapter(hidden_size=4096, rank=512, use_initializer=True)
    assert trainable_parameter_count(adapter) / 6_738_415_616 < 0.01
