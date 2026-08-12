from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def test_local_teacher_is_closer_to_endpoint():
    source = (ROOT / "train_transformer.py").read_text()
    assert "out_tn_1 = cd_ema(x_tn_1, x_tn, tn_1.unsqueeze(0)" in source
    assert "cd(x_tn, x_tn_prev, tn.unsqueeze(0)" in source


try:
    import torch
except ModuleNotFoundError:
    test_local_teacher_is_closer_to_endpoint()
    print("cdeq_time_boundary=skipped_missing_torch")
    raise SystemExit(0)

for path in (ROOT, ROOT.parent):
    sys.path.insert(0, str(path))

from models.deq_transformer_CD import cm_boundary_mix


def test_cm_boundary_mix():
    z = torch.full((2, 1, 3, 4), 7.0)
    out = torch.full_like(z, 11.0)

    assert torch.equal(cm_boundary_mix(z, out, torch.zeros(2, 1)), out)
    assert torch.equal(cm_boundary_mix(z, out, torch.full((2, 1), 5.0)), z)


if __name__ == "__main__":
    test_cm_boundary_mix()
    test_local_teacher_is_closer_to_endpoint()
    print("cdeq_time_boundary=ok")
