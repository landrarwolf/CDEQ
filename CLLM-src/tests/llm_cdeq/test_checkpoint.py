import torch

from llm_cdeq.model import AdapterMetrics, CDEQAdapter, make_ema
from llm_cdeq.train import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    config = {
        "schema_version": "llm_cdeq_config_v1",
        "upstream": {"code_revision": "abc"},
        "model": {"hidden_size": 8, "bottleneck_rank": 2},
    }
    adapter = CDEQAdapter(8, 2, use_initializer=True)
    ema = make_ema(adapter)
    optimizer = torch.optim.AdamW(adapter.consistency_parameters())
    init_optimizer = torch.optim.AdamW(adapter.initializer_parameters())
    metrics = AdapterMetrics(0.2, 0.5, 0.1, 4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        adapter,
        ema,
        optimizer,
        init_optimizer,
        config,
        metrics,
        epoch=2,
        global_step=7,
        use_ct=True,
        train_manifest={"data_split_hash": "split", "backbone_parameter_count": 1000},
    )
    restored = CDEQAdapter(8, 2, use_initializer=True)
    package = load_checkpoint(path, restored)
    assert package["schema_version"] == "llm_cdeq_checkpoint_v1"
    assert package["best_validation_metrics"]["endpoint_relative_error"] == 0.2
    for expected, actual in zip(adapter.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
