import pytest
import torch

from llm_cdeq.model import CDEQAdapter, make_ema
from llm_cdeq.time import rho_time_grid
from llm_cdeq.train import _interpolate_batch, train_step


def make_batch():
    base = torch.randn(4, 2, 8)
    endpoint = torch.randn(4, 2, 8)
    states = torch.stack((base, 0.5 * (base + endpoint), endpoint), dim=1)
    times = rho_time_grid(3).expand(4, -1).clone()
    return {
        "states": states,
        "state_mask": torch.ones(4, 3, dtype=torch.bool),
        "time_grid": times,
        "token_mask": torch.ones(4, 2, dtype=torch.bool),
        "endpoint_tokens": torch.zeros(4, 2, dtype=torch.long),
        "trajectory_tokens": torch.zeros(4, 3, 2, dtype=torch.long),
    }


def test_batched_interpolation_respects_per_sample_trajectory_length():
    states = torch.tensor(
        [
            [[[0.0]], [[10.0]], [[20.0]]],
            [[[0.0]], [[30.0]], [[999.0]]],
        ]
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])
    times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, float("nan")]])
    result = _interpolate_batch(states, mask, times, torch.tensor([1.5, 1.5]))
    torch.testing.assert_close(result.flatten(), torch.tensor([15.0, 15.0]))


@pytest.mark.parametrize("use_initializer", [False, True])
@pytest.mark.parametrize("use_ct", [False, True])
def test_all_ablation_training_steps_are_finite(use_initializer, use_ct):
    adapter = CDEQAdapter(8, 2, use_initializer=use_initializer)
    ema = make_ema(adapter)
    config = {
        "time": {"epsilon": 0.002, "terminal": 5.0, "q": 1.1, "d": 100, "k": 8, "b": 1},
        "training": {"local_weight": 0.1, "endpoint_weight": 0.9, "token_ce_weight": 0.0},
    }
    loss, parts = train_step(
        adapter,
        ema,
        make_batch(),
        config,
        use_ct=use_ct,
        global_step=200,
        generator=torch.Generator().manual_seed(42),
        lm_head_weight=None,
    )
    assert torch.isfinite(loss)
    assert parts["local"] >= 0
    assert parts["endpoint"] >= 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in adapter.consistency_parameters())
