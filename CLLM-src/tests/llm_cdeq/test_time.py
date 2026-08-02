import pytest
import torch

from llm_cdeq.time import interpolate_trajectory, rho_time_grid, sample_continuous_pair


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_rho_grid_has_exact_typed_endpoints_and_is_monotone(dtype):
    epsilon = torch.tensor(0.002, dtype=dtype)
    terminal = torch.tensor(5.0, dtype=dtype)
    for steps in range(2, 18):
        grid = rho_time_grid(
            steps,
            epsilon=epsilon.item(),
            terminal=terminal.item(),
            rho=7.0,
            dtype=dtype,
        )
        assert torch.equal(grid[0], epsilon)
        assert torch.equal(grid[-1], terminal)
        assert torch.all(grid[1:] > grid[:-1])


def test_interpolation_only_changes_hidden_values():
    states = torch.tensor([[0.0, 2.0], [10.0, 12.0], [20.0, 22.0]])
    times = torch.tensor([0.0, 1.0, 2.0])
    result = interpolate_trajectory(states, times, torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(result, torch.tensor([[5.0, 7.0], [15.0, 17.0]]))


def test_progressive_pair_respects_domain_and_order():
    generator = torch.Generator().manual_seed(3)
    later, earlier = sample_continuous_pair(512, 500, generator=generator)
    assert torch.all(earlier >= 0.002)
    assert torch.all(later <= 5.0)
    assert torch.all(earlier <= later)
