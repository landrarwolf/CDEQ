from __future__ import annotations

import torch


def rho_time_grid(
    steps: int,
    *,
    epsilon: float = 0.002,
    terminal: float = 5.0,
    rho: float = 7.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0 < epsilon < terminal:
        raise ValueError("expected 0 < epsilon < terminal")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if steps == 1:
        return torch.tensor([terminal], device=device, dtype=dtype)
    j = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    start = torch.as_tensor(epsilon, device=device, dtype=dtype).pow(1 / rho)
    end = torch.as_tensor(terminal, device=device, dtype=dtype).pow(1 / rho)
    return (start + j * (end - start)).pow(rho)


def interpolate_trajectory(
    states: torch.Tensor,
    times: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate `[K, ...]` states at one or more query times."""
    if states.ndim < 2:
        raise ValueError("states must have shape [K, ...]")
    if times.ndim != 1 or times.numel() != states.shape[0]:
        raise ValueError("times must be one-dimensional and match states.shape[0]")
    if times.numel() < 2:
        raise ValueError("at least two trajectory states are required")
    query = torch.as_tensor(query, device=states.device, dtype=times.dtype).flatten()
    times = times.to(device=states.device)
    right = torch.searchsorted(times, query).clamp(1, times.numel() - 1)
    left = right - 1
    denom = (times[right] - times[left]).clamp_min(torch.finfo(times.dtype).eps)
    weight = (query - times[left]) / denom
    view_shape = (query.numel(),) + (1,) * (states.ndim - 1)
    return states[left] * (1 - weight.view(view_shape)) + states[right] * weight.view(view_shape)


def sample_continuous_pair(
    batch_size: int,
    global_step: int,
    *,
    epsilon: float = 0.002,
    terminal: float = 5.0,
    q: float = 1.1,
    d: int = 100,
    k: float = 8.0,
    b: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if q <= 1 or d <= 0:
        raise ValueError("q must be > 1 and d must be positive")
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    eps = torch.as_tensor(epsilon, device=device, dtype=dtype)
    end = torch.as_tensor(terminal, device=device, dtype=dtype)
    t = torch.exp(torch.log(eps) + u * (torch.log(end) - torch.log(eps)))
    n_t = 1 + k * torch.sigmoid(-b * t)
    q_power = q ** (global_step // d)
    r = t * (1 - n_t / q_power)
    r = torch.minimum(r.clamp_min(eps), t)
    return t, r

