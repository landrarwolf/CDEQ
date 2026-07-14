from llm_cdeq.analyze import feasibility_summary


def test_feasibility_gate_accepts_positive_components_and_best_combination():
    rows = [
        {
            "init": 0,
            "ct": 0,
            "endpoint_relative_error": 1.0,
            "token_agreement": 0.20,
            "initial_endpoint_relative_error": 1.30,
            "initial_token_agreement": 0.10,
        },
        {"init": 1, "ct": 0, "endpoint_relative_error": 0.94, "token_agreement": 0.20},
        {"init": 0, "ct": 1, "endpoint_relative_error": 1.0, "token_agreement": 0.23},
        {"init": 1, "ct": 1, "endpoint_relative_error": 0.90, "token_agreement": 0.24},
    ]
    summary = feasibility_summary(rows)
    assert summary["passes_single_seed_direction_gate"]
    assert summary["baseline_vs_identity"]["passes"]
    assert summary["passes_single_seed_feasibility_gate"]
