from pathlib import Path
import sys

try:
    import torch
except ModuleNotFoundError:
    print("cdeq_time_boundary=skipped_missing_torch")
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT.parent):
    sys.path.insert(0, str(path))

from models.deq_transformer_CD import (
    InitialStatePredictor,
    cm_boundary_mix,
    interpolate_trajectory,
    sample_continuous_pair,
)


def test_cm_boundary_mix():
    z = torch.full((2, 1, 3, 4), 7.0)
    out = torch.full_like(z, 11.0)

    assert torch.equal(cm_boundary_mix(z, out, torch.zeros(2, 1)), out)
    assert torch.equal(cm_boundary_mix(z, out, torch.full((2, 1), 5.0)), z)


def test_continuous_pair_and_interpolation():
    t_traj = torch.tensor([0.002, 1.0, 5.0])
    x_traj = torch.arange(2 * 3 * 1 * 1, dtype=torch.float32).view(2, 3, 1, 1)
    t, r = sample_continuous_pair(t_traj, n_steps=8, global_step=0)

    assert torch.all(t >= t_traj[0])
    assert torch.all(t <= t_traj[-1])
    assert torch.all(r >= t_traj[0])
    assert torch.all(r <= t)
    assert torch.equal(interpolate_trajectory(x_traj, t_traj, t_traj[:1]), x_traj[:, :1])
    assert torch.equal(interpolate_trajectory(x_traj, t_traj, t_traj[-1:]), x_traj[:, -1:])


def test_initial_state_predictor_backward():
    model = InitialStatePredictor(d_model=4)
    us = torch.randn(2, 12, 5)
    target = torch.randn(2, 4, 5)
    out = model(us)
    loss = torch.nn.functional.smooth_l1_loss(out, target)
    loss.backward()

    assert out.shape == target.shape
    assert all(param.grad is not None for param in model.parameters())


if __name__ == "__main__":
    test_cm_boundary_mix()
    test_continuous_pair_and_interpolation()
    test_initial_state_predictor_backward()
    print("cdeq_time_boundary=ok")
