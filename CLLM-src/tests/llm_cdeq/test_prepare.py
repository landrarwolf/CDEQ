import torch

from llm_cdeq.prepare_states import (
    deduplicate_token_states,
    plan_grouped_split,
    recover_aligned_chain,
    shifted_hidden_slice,
)


def test_hidden_shift_starts_at_last_prompt_position():
    hidden = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    shifted = shifted_hidden_slice(hidden, prompt_length=4, block_size=3)
    torch.testing.assert_close(shifted.flatten(), torch.tensor([3.0, 4.0, 5.0]))


def test_interleaved_augmented_states_recover_longest_jacobi_chain():
    # Candidate JSON order is deliberately unrelated to solver time.
    candidates = torch.tensor([[3, 3], [0, 0], [2, 2], [1, 1]])
    predictions = torch.tensor([[3, 3], [1, 1], [3, 3], [2, 2]])
    chain = recover_aligned_chain(
        candidates,
        predictions,
        torch.tensor([3, 3]),
        eos_token_id=None,
        max_states=4,
    )
    assert chain == [1, 3, 2, 0]


def test_deduplication_ignores_tokens_after_eos():
    states = deduplicate_token_states([[1, 9, 2], [1, 9, 7], [1, 8, 2]], eos_token_id=9)
    assert states.tolist() == [[1, 9, 2], [1, 8, 2]]


def test_grouped_split_preserves_group_diversity_without_leaking_groups():
    counts = {"large": 90, **{f"small-{index}": 1 for index in range(30)}}
    assignment, summary = plan_grouped_split(
        counts,
        seed=42,
        validation_fraction=0.1,
        validation_minimum=12,
    )
    assert summary["validation_records"] >= 12
    assert summary["validation_groups"] >= 3
    assert summary["train_groups"] >= 18
    assert summary["validation_group_target"] == 3
    assert summary["train_records"] + summary["validation_records"] == 120
    assert set(assignment) == set(counts)
    assert set(assignment.values()) == {"train", "validation"}
    repeated, repeated_summary = plan_grouped_split(
        counts,
        seed=42,
        validation_fraction=0.1,
        validation_minimum=12,
    )
    assert repeated == assignment
    assert repeated_summary == summary
