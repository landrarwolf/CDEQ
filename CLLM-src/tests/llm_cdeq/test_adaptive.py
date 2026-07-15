import pytest
import torch

from llm_cdeq.adaptive import (
    adaptive_oracle_recurrence,
    progress_to_rho_time,
    project_to_teacher_trajectory,
    rho_time_to_progress,
)


class RecordingAdapter:
    terminal = 5.0

    def __init__(self, increments):
        self.increments = list(increments)
        self.initialize_calls = 0
        self.consistency_calls = 0
        self.times = []

    def initialize(self, state):
        self.initialize_calls += 1
        return state + 1

    def consistency(self, state, time):
        self.times.append(time.clone())
        increment = self.increments[min(self.consistency_calls, len(self.increments) - 1)]
        self.consistency_calls += 1
        scale = (time < self.terminal).to(state.dtype)[:, None, None]
        return state + scale * increment


def line_trajectory(values):
    return torch.tensor(values, dtype=torch.float32)[:, None, None]


def test_rho_progress_mapping_round_trip_and_endpoints():
    progress = torch.tensor([0.0, 0.25, 1.0])
    time = progress_to_rho_time(progress)
    assert time[0].item() == pytest.approx(0.002, rel=1e-5)
    assert time[-1].item() == pytest.approx(5.0, rel=1e-5)
    torch.testing.assert_close(rho_time_to_progress(time), progress, atol=2e-6, rtol=2e-6)


def test_continuous_oracle_projects_between_points_and_reports_margin():
    trajectory = line_trajectory([1.0, 3.0, 7.0])
    result = project_to_teacher_trajectory(
        torch.tensor([[2.0]]),
        trajectory,
        torch.tensor([True, True, True]),
        torch.tensor([True]),
    )
    assert result.valid
    assert result.segment_index.item() == 0
    assert result.segment_fraction.item() == pytest.approx(0.5)
    assert result.progress.item() == pytest.approx(0.25)
    assert result.distance.item() == pytest.approx(0.0)
    assert result.second_distance.item() > result.distance.item()
    assert result.margin.item() == pytest.approx(
        result.second_distance.item() - result.distance.item()
    )


def test_projection_ignores_padded_states_and_masked_eos_tail():
    trajectory = torch.tensor(
        [
            [[1.0], [10.0]],
            [[3.0], [20.0]],
            [[5.0], [30.0]],
            [[1000.0], [-1000.0]],
        ]
    )
    query = torch.tensor([[4.0], [-9999.0]])
    result = project_to_teacher_trajectory(
        query,
        trajectory,
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False]),
    )
    assert result.segment_index.item() == 1
    assert result.segment_fraction.item() == pytest.approx(0.5)
    assert result.progress.item() == pytest.approx(0.75)
    assert result.distance.item() == pytest.approx(0.0)


def test_projection_handles_variable_length_batched_trajectories():
    trajectories = torch.tensor(
        [
            [[[1.0]], [[3.0]], [[5.0]], [[999.0]]],
            [[[2.0]], [[4.0]], [[6.0]], [[8.0]]],
        ]
    )
    result = project_to_teacher_trajectory(
        torch.tensor([[[4.0]], [[7.0]]]),
        trajectories,
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
        torch.ones(2, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(result.progress, torch.tensor([0.75, 5 / 6]))
    assert result.segment_index.tolist() == [1, 2]


def test_projection_distance_is_normalized_by_teacher_scale():
    first = project_to_teacher_trajectory(
        torch.tensor([[3.0]]), line_trajectory([1.0, 2.0]), torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool)
    )
    second = project_to_teacher_trajectory(
        torch.tensor([[30.0]]), line_trajectory([10.0, 20.0]), torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool)
    )
    assert first.distance.item() == pytest.approx(0.5)
    assert second.distance.item() == pytest.approx(0.5)


def test_recurrence_applies_initializer_once_and_uses_oracle_time_after_zero():
    adapter = RecordingAdapter([2.0, 2.0, 2.0, 2.0])
    result = adaptive_oracle_recurrence(
        adapter,
        torch.zeros(1, 1, 1),
        line_trajectory([0.0, 2.0, 4.0, 6.0, 8.0]).unsqueeze(0),
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        max_calls=3,
        endpoint_distance=0.0,
    )
    assert adapter.initialize_calls == 1
    assert adapter.consistency_calls == 3
    assert adapter.times[0].item() == 0.0
    assert 0.002 < adapter.times[1].item() < adapter.terminal
    assert adapter.times[2].item() > adapter.times[1].item()
    assert result.calls.item() == 3
    assert result.stop_reasons == ("budget",)


def test_unverified_endpoint_is_capped_below_identity_boundary():
    adapter = RecordingAdapter([10.0, 0.0])
    result = adaptive_oracle_recurrence(
        adapter,
        torch.zeros(1, 1, 1),
        line_trajectory([1.0, 5.0, 10.0]).unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        max_calls=2,
        endpoint_distance=0.0,
        identity_cap_progress=0.9,
    )
    assert adapter.times[1].item() < adapter.terminal
    assert result.steps[0].projection.progress.item() == pytest.approx(1.0)


def test_recurrence_stops_on_clear_progress_regression():
    adapter = RecordingAdapter([5.0, -5.0, 1.0])
    result = adaptive_oracle_recurrence(
        adapter,
        torch.zeros(1, 1, 1),
        line_trajectory([1.0, 3.0, 5.0, 7.0]).unsqueeze(0),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        max_calls=4,
        endpoint_distance=0.0,
        regression_tolerance=0.1,
    )
    assert result.calls.item() == 2
    assert result.stop_reasons == ("regression",)


def test_exact_endpoint_stops_without_an_identity_call():
    adapter = RecordingAdapter([2.0, 10.0])
    result = adaptive_oracle_recurrence(
        adapter,
        torch.zeros(1, 1, 1),
        line_trajectory([1.0, 2.0, 3.0]).unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        max_calls=4,
    )
    assert result.calls.item() == 1
    assert result.stop_reasons == ("endpoint",)
    assert adapter.consistency_calls == 1


def test_batched_recurrence_freezes_finished_examples_at_identity_time():
    adapter = RecordingAdapter([2.0, 2.0])
    trajectories = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]]],
            [[[1.0]], [[3.0]], [[5.0]]],
        ]
    )
    result = adaptive_oracle_recurrence(
        adapter,
        torch.zeros(2, 1, 1),
        trajectories,
        torch.ones(2, 3, dtype=torch.bool),
        torch.ones(2, 1, dtype=torch.bool),
        max_calls=4,
    )
    assert result.calls.tolist() == [1, 2]
    assert result.stop_reasons == ("endpoint", "endpoint")
    assert adapter.times[1][0].item() == adapter.terminal
    assert adapter.times[1][1].item() < adapter.terminal
    torch.testing.assert_close(result.state.flatten(), torch.tensor([3.0, 5.0]))
