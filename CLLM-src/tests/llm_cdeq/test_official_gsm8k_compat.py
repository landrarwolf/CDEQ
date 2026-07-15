import importlib.util
import sys
from pathlib import Path


def load_acc_module():
    root = Path(__file__).resolve().parents[2]
    gsm8k_dir = root / "eval" / "gsm8k"
    sys.path.insert(0, str(gsm8k_dir))
    spec = importlib.util.spec_from_file_location("cllm_gsm8k_acc", gsm8k_dir / "acc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gsm8k_data_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    module = load_acc_module()
    monkeypatch.chdir(tmp_path)
    prompts = module.get_raw_inputs("gsm8k")
    assert len(prompts) == 1319
    assert prompts[0]
