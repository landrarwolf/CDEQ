from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def test_ea_pit_training_source():
    source = (ROOT / "train_transformer.py").read_text()
    unified = source.split(
        "optimizer.zero_grad()\n                if args.cm_continuous_time:", 1
    )[1].split("\n                else:", 1)[0]
    discrete = source.split(
        "optimizer.zero_grad()\n                if args.cm_continuous_time:", 1
    )[1].split("\n                else:", 1)[1].split("\n\n                loss.backward()", 1)[0]

    assert "r, s, alpha = sample_ea_pit_pair(" in source
    assert "target = cd_ema(\n                            z_s," in unified
    assert "prediction = cd(\n                        z_r," in unified
    assert unified.count("F.smooth_l1_loss(") == 1
    assert "F.mse_loss(" not in unified
    assert "prediction = torch.cat((prediction, init_prediction), dim=1)" in unified
    assert "target = torch.cat((target, x_endpoint.unsqueeze(1).detach()), dim=1)" in unified
    assert "loss = F.smooth_l1_loss(prediction, target)" in unified
    assert "loss = 0.1 * loss_1 + 0.9 * global_loss" in discrete
    assert "init_loss.backward()" in source
    assert "init_optimizer.step()" in source
    assert "z_init = init_model" in source and ".detach()" in source


def test_ea_pit_checkpoint_source():
    source = (ROOT / "train_transformer.py").read_text()
    for field in (
        '"cm_schedule_version"',
        '"cm_ct_params"',
        '"cm_continuous_time"',
        '"cdeq_init"',
        '"cm_global_step"',
        '"best_rel_diff"',
    ):
        assert field in source
    validation = source.index("        validate_cm_checkpoint_config(checkpoint, args)")
    optimizer_load = source.index('optimizer.load_state_dict(checkpoint["optimizer"])')
    assert validation < optimizer_load
    assert 'best_rel_diff = checkpoint.get("best_rel_diff", float("inf"))' in source


try:
    import torch
except ModuleNotFoundError:
    test_ea_pit_training_source()
    test_ea_pit_checkpoint_source()
    print("cdeq_time_boundary=skipped_missing_torch")
    raise SystemExit(0)

for path in (ROOT, ROOT.parent):
    sys.path.insert(0, str(path))

from models.deq_transformer_CD import (
    InitialStatePredictor,
    cm_boundary_mix,
    interpolate_trajectory,
    sample_ea_pit_pair,
)
from train_transformer import (
    CM_SCHEDULE_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    load_cm_package,
    save_cm_package,
    validate_cm_checkpoint_config,
)


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


def _s_min(t_traj, r):
    right = torch.searchsorted(t_traj, r, right=True).clamp_max(t_traj.numel() - 1)
    return t_traj[right]


def _assert_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _args(continuous=True, cdeq_init=False, p_end=0.1):
    return SimpleNamespace(
        cm_continuous_time=continuous,
        cdeq_init=cdeq_init,
        cdeq_init_lr=1e-4,
        cdeq_init_steps=10,
        cm_ct_q=1.1,
        cm_ct_d=100,
        cm_ct_k=8.0,
        cm_ct_b=1.0,
        cm_ct_p_end=p_end,
    )


def _checkpoint():
    return {
        "cm_continuous_time": True,
        "cm_schedule_version": CM_SCHEDULE_VERSION,
        "cm_global_step": 123,
        "best_rel_diff": 0.25,
        "cm_ct_params": {"q": 1.1, "d": 100, "k": 8.0, "b": 1.0, "p_end": 0.1},
        "cdeq_init": False,
    }


def test_cm_boundary_mix():
    z = torch.full((2, 1, 3, 4), 7.0)
    out = torch.full_like(z, 11.0)

    assert torch.equal(cm_boundary_mix(z, out, torch.zeros(2, 1)), out)
    assert torch.equal(cm_boundary_mix(z, out, torch.full((2, 1), 5.0)), z)


def test_ea_pit_formula_and_interpolation():
    t_traj = torch.tensor([0.002, 0.1, 1.0, 5.0])
    x_traj = torch.arange(2 * 4, dtype=torch.float32).view(2, 4, 1, 1)
    r, s, alpha = sample_ea_pit_pair(
        t_traj, 512, global_step=3000, q=1.1, d=100, k=8, b=1,
        p_end=0, generator=_generator(7),
    )
    s_min = _s_min(t_traj, r)
    expected_alpha = (
        (1 + 8 * torch.sigmoid(-(t_traj[-1] - r))) * (1.1 ** -30)
    ).clamp_max(1)
    expected_s = s_min + expected_alpha * (t_traj[-1] - s_min)

    assert torch.allclose(alpha, expected_alpha)
    assert torch.allclose(s, expected_s)
    assert torch.all(r < s_min)
    assert torch.all(s_min <= s)
    assert torch.all(s <= t_traj[-1])
    assert torch.all((0 <= alpha) & (alpha <= 1))
    assert torch.equal(interpolate_trajectory(x_traj, t_traj, t_traj[:1]), x_traj[:, :1])
    assert torch.equal(interpolate_trajectory(x_traj, t_traj, t_traj[-1:]), x_traj[:, -1:])


def test_ea_pit_starts_at_endpoint():
    t_traj = torch.tensor([0.002, 0.1, 1.0, 5.0])
    x_traj = torch.arange(2 * 4, dtype=torch.float32).view(2, 4, 1, 1)
    r, s, alpha = sample_ea_pit_pair(
        t_traj, 64, global_step=0, p_end=0, generator=_generator(1)
    )
    z_s = interpolate_trajectory(x_traj, t_traj, s)

    assert torch.all(alpha == 1)
    assert torch.all(s == t_traj[-1])
    assert torch.equal(z_s, x_traj[:, -1:].expand(-1, s.numel(), -1, -1))
    assert torch.all(r < _s_min(t_traj, r))


def test_ea_pit_tightens_at_stage_boundary():
    t_traj = torch.tensor([0.002, 0.1, 1.0, 5.0])
    results = [
        sample_ea_pit_pair(
            t_traj, 128, step, q=2, d=10, k=0, p_end=0,
            generator=_generator(11),
        )
        for step in (0, 9, 10, 20)
    ]
    r0, s0, alpha0 = results[0]

    assert all(torch.equal(r0, r) for r, _, _ in results[1:])
    assert torch.equal(alpha0, results[1][2])
    assert torch.all(alpha0 == 1)
    assert torch.allclose(results[2][2], torch.full_like(alpha0, 0.5))
    assert torch.allclose(results[3][2], torch.full_like(alpha0, 0.25))
    assert torch.all(results[2][1] <= s0)
    assert torch.all(results[3][1] <= results[2][1])


def test_ea_pit_samples_solver_intervals_uniformly():
    t_traj = torch.tensor([0.002, 0.003, 0.1, 5.0])
    r, _, _ = sample_ea_pit_pair(
        t_traj, 6000, global_step=100000, p_end=0, generator=_generator(19)
    )
    interval = torch.searchsorted(t_traj, r, right=True) - 1
    counts = torch.bincount(interval, minlength=3)

    assert torch.all((counts > 1800) & (counts < 2200)), counts


def test_ea_pit_endpoint_anchoring():
    t_traj = torch.tensor([0.002, 0.1, 1.0, 3.0, 5.0])
    samples = {
        p_end: sample_ea_pit_pair(
            t_traj, 10000, global_step=100000, p_end=p_end,
            generator=_generator(23),
        )[1]
        for p_end in (0, 0.1, 1)
    }
    rates = {p_end: (s == t_traj[-1]).float().mean().item() for p_end, s in samples.items()}

    assert 0.20 < rates[0] < 0.30
    assert abs(rates[0.1] - 0.325) < 0.025
    assert rates[1] == 1


def test_ea_pit_large_step_and_nextafter_are_safe():
    first = torch.tensor(0.002)
    adjacent = torch.nextafter(first, torch.tensor(float("inf")))
    t_traj = torch.stack((first, adjacent, torch.tensor(5.0)))
    r, s, alpha = sample_ea_pit_pair(
        t_traj, 2048, global_step=10 ** 12, p_end=0, generator=_generator(29)
    )
    s_min = _s_min(t_traj, r)

    assert torch.isfinite(r).all() and torch.isfinite(s).all() and torch.isfinite(alpha).all()
    assert torch.all(r < s_min)
    assert torch.equal(s, s_min)
    assert torch.all(alpha == 0)


def test_ea_pit_rejects_invalid_inputs():
    invalid_grids = (
        torch.tensor([0.002]),
        torch.tensor([[0.002, 5.0]]),
        torch.tensor([0.002, 1.0, 1.0, 5.0]),
        torch.tensor([0.002, float("nan"), 5.0]),
        torch.tensor([0.002, 4.9]),
        torch.tensor([0, 5]),
    )
    for t_traj in invalid_grids:
        _assert_value_error(lambda t_traj=t_traj: sample_ea_pit_pair(t_traj, 1, 0))

    valid = torch.tensor([0.002, 1.0, 5.0])
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 0, 0))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1.5, 0))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, -1))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0.5))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, q=1))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, d=0.5))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, p_end=1.1))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, q=float("nan")))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, k=float("inf")))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, b=float("nan")))
    _assert_value_error(lambda: sample_ea_pit_pair(valid, 1, 0, p_end=float("nan")))


def test_ea_pit_checkpoint_contract():
    checkpoint = _checkpoint()
    validate_cm_checkpoint_config(checkpoint, _args())
    validate_cm_checkpoint_config({"cm_time_convention": TRAJECTORY_FORMAT_VERSION}, _args(False))
    legacy_init = {"init_model": {}, "init_optimizer": {}}
    validate_cm_checkpoint_config(legacy_init, _args(False, cdeq_init=True))
    _assert_value_error(lambda: validate_cm_checkpoint_config(legacy_init, _args(False)))

    for field in ("cm_global_step", "best_rel_diff", "cdeq_init"):
        invalid = checkpoint.copy()
        invalid.pop(field)
        _assert_value_error(lambda invalid=invalid: validate_cm_checkpoint_config(invalid, _args()))
    for field, value in (
        ("cm_global_step", -1),
        ("cm_global_step", 1.5),
        ("best_rel_diff", float("inf")),
    ):
        invalid = checkpoint.copy()
        invalid[field] = value
        _assert_value_error(lambda invalid=invalid: validate_cm_checkpoint_config(invalid, _args()))

    _assert_value_error(lambda: validate_cm_checkpoint_config(checkpoint, _args(p_end=0.2)))
    _assert_value_error(lambda: validate_cm_checkpoint_config(checkpoint, _args(False)))
    _assert_value_error(lambda: validate_cm_checkpoint_config(checkpoint, _args(cdeq_init=True)))


def test_cm_package_contract_and_legacy_load():
    model = torch.nn.Linear(2, 2)
    with tempfile.TemporaryDirectory() as directory:
        ea_path = Path(directory) / "ea_pit.pth"
        off_path = Path(directory) / "ordinary.pth"
        legacy_path = Path(directory) / "legacy.pth"
        save_cm_package(
            ea_path, model, args=_args(), cm_global_step=7, best_rel_diff=0.5
        )
        save_cm_package(
            off_path, model, args=_args(False), cm_global_step=3, best_rel_diff=0.75
        )
        torch.save({"weight": torch.ones(1)}, legacy_path)

        ea = torch.load(ea_path, map_location="cpu")
        ordinary = torch.load(off_path, map_location="cpu")
        legacy = load_cm_package(legacy_path, torch.device("cpu"))

    assert ea["cm_schedule_version"] == CM_SCHEDULE_VERSION
    assert ea["cm_global_step"] == 7 and ea["best_rel_diff"] == 0.5
    assert ea["cm_ct_params"]["p_end"] == 0.1
    assert ordinary["cm_method"] == "cdeq"
    assert ordinary["cm_continuous_time"] is False
    assert ordinary["cm_schedule_version"] is None
    assert torch.equal(legacy["model"]["weight"], torch.ones(1))


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
    test_ea_pit_formula_and_interpolation()
    test_ea_pit_starts_at_endpoint()
    test_ea_pit_tightens_at_stage_boundary()
    test_ea_pit_samples_solver_intervals_uniformly()
    test_ea_pit_endpoint_anchoring()
    test_ea_pit_large_step_and_nextafter_are_safe()
    test_ea_pit_rejects_invalid_inputs()
    test_ea_pit_checkpoint_contract()
    test_cm_package_contract_and_legacy_load()
    test_initial_state_predictor_backward()
    test_ea_pit_training_source()
    test_ea_pit_checkpoint_source()
    print("cdeq_time_boundary=ok")
