from __future__ import annotations

import math

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.evaluation.search_benchmark_utils import (
    confusion_ratio_vs_oracle,
    finite_mean,
    normalized_confusion_ratio,
    oracle_status_from_outcome,
    path_efficiency_ratio,
    path_transition_count,
    run_astar_oracle,
)
from src.simulation.validator import ZeldaLogicEnv
from src.simulation.search_status import (
    proven_unsolvable_from_outcome,
    search_outcome_is_indeterminate,
    search_proof_status,
)
from src.evaluation.pcbs_telemetry_calibration import _path_efficiency


def _simple_grid() -> np.ndarray:
    grid = np.full((6, 8), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int64)
    grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[4, 6] = int(SEMANTIC_PALETTE["TRIFORCE"])
    return grid


def test_oracle_status_normalizes_common_failure_modes():
    assert oracle_status_from_outcome(True, "") == "solved"
    assert oracle_status_from_outcome(False, "Timeout: explored 100 states") == "failed"
    assert oracle_status_from_outcome(False, "No path: all reachable states explored") == "failed"
    assert oracle_status_from_outcome(False, "No goal (TRIFORCE) found in map") == "failed"
    assert oracle_status_from_outcome(False, "misc error") == "failed"
    assert oracle_status_from_outcome(False, "localized message", "budget_exhausted") == "timeout"
    assert oracle_status_from_outcome(False, "localized message", "exhausted") == "no_path"
    assert oracle_status_from_outcome(False, "localized message", "invalid") == "invalid_map"
    assert oracle_status_from_outcome(False, "localized message", "depth_limit") == "inconclusive"


def test_search_proof_requires_structured_termination_evidence():
    assert search_proof_status(True, "solved") == "solution_witness"
    assert search_proof_status(False, "exhausted") == "no_path_proof"
    assert proven_unsolvable_from_outcome(False, "exhausted") is True
    for status in ("budget_exhausted", "depth_limit", "incomplete", "route_replay_failed"):
        assert proven_unsolvable_from_outcome(False, status) is False
        assert search_outcome_is_indeterminate(False, status) is True
    assert search_proof_status(True, "budget_exhausted") == "inconsistent"


def test_path_efficiency_ratio_is_bounded_and_directionally_consistent():
    assert path_efficiency_ratio(10, 8) == 0.8
    assert path_efficiency_ratio(0, 8) == 0.0
    assert path_efficiency_ratio(8, 0) == 0.0


def test_path_transition_count_excludes_the_starting_state():
    assert path_transition_count([]) == 0
    assert path_transition_count([(1, 1)]) == 0
    assert path_transition_count([(1, 1), (1, 2), (2, 2)]) == 2


def test_telemetry_path_efficiency_uses_bounded_oracle_over_candidate_ratio():
    assert _path_efficiency({"path_length": 20, "oracle_path_length": 10}) == 0.5
    assert _path_efficiency({"path_length": 5, "oracle_path_length": 10}) == 1.0
    assert _path_efficiency({"path_efficiency": 1.5}) == 1.0


def test_ab_benchmark_is_import_safe_and_headless():
    from scripts import ab_benchmark

    assert callable(ab_benchmark.main)
    assert "gui_runner" not in ab_benchmark.__dict__


def test_confusion_ratio_vs_oracle_returns_nan_when_oracle_not_resolved():
    assert math.isnan(confusion_ratio_vs_oracle(12, 20, oracle_status="timeout", candidate_success=True))
    assert math.isnan(confusion_ratio_vs_oracle(12, 20, oracle_status="solved", candidate_success=False))
    assert confusion_ratio_vs_oracle(10, 15, oracle_status="solved", candidate_success=True) == 0.5


def test_normalized_confusion_ratio_uses_excess_path_not_raw_ratio():
    assert normalized_confusion_ratio(2, 4, 2, oracle_status="solved", candidate_success=True) == 1.0
    assert normalized_confusion_ratio(200, 400, 2, oracle_status="solved", candidate_success=True) == 1.0
    assert normalized_confusion_ratio(1, 100, 1, oracle_status="solved", candidate_success=True) == 1.0
    assert math.isnan(normalized_confusion_ratio(0, 4, 2, oracle_status="timeout", candidate_success=True))


def test_finite_mean_ignores_nan_and_inf():
    assert finite_mean([1.0, float("nan"), float("inf"), 3.0]) == 2.0
    assert finite_mean([float("nan"), float("inf")]) == 0.0


def test_run_astar_oracle_reports_solved_status_on_simple_grid():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    payload = run_astar_oracle(env, timeout=20000)
    assert payload["success"] is True
    assert payload["status"] == "solved"
    assert payload["path_length"] > 0
    assert payload["states_explored"] > 0
    assert payload["states_explored"] <= payload["state_budget"]
    assert payload["termination_status"] == "solved"
    assert payload["proof_status"] == "solution_witness"
    assert payload["route_replay_status"] == "verified"
    assert payload["proven_unsolvable"] is False
    assert payload["indeterminate"] is False


def test_astar_frontier_exhaustion_is_not_rerun_as_dijkstra() -> None:
    grid = _simple_grid()
    grid[2, 1:-1] = int(SEMANTIC_PALETTE["WALL"])
    env = ZeldaLogicEnv(grid, render_mode=False)

    payload = run_astar_oracle(env, timeout=40)

    assert payload["success"] is False
    assert payload["states_explored"] <= 40
    assert payload["termination_status"] == "exhausted"
    assert payload["proof_status"] == "no_path_proof"
    assert payload["proven_unsolvable"] is True
    assert payload["indeterminate"] is False
    assert payload["fallback_attempted"] is False
    assert payload["fallback_states_explored"] == 0


def test_astar_oracle_rejects_an_unreplayable_solution(monkeypatch) -> None:
    from src.evaluation import search_benchmark_utils as module

    monkeypatch.setattr(
        module.StateSpaceAStar,
        "verify_position_path",
        lambda *_args, **_kwargs: (False, "synthetic replay failure", None, None),
    )

    payload = run_astar_oracle(
        ZeldaLogicEnv(_simple_grid(), render_mode=False),
        timeout=20_000,
    )

    assert payload["success"] is False
    assert payload["termination_status"] == "route_replay_failed"
    assert payload["proof_status"] == "execution_error"
    assert payload["proven_unsolvable"] is False
    assert payload["indeterminate"] is True
    assert payload["route_replay_status"] == "failed"


def test_cbs_fitness_rejects_timeout_without_running_pcbs(monkeypatch) -> None:
    from src.evaluation import cbs_fitness as cbs_fitness_module

    monkeypatch.setattr(
        cbs_fitness_module,
        "run_astar_oracle",
        lambda *_args, **_kwargs: {
            "success": False,
            "path": [],
            "states_explored": 25,
            "status": "timeout",
            "failure_reason": "state budget exhausted",
        },
    )

    class _UnexpectedPCBS:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("P-CBS must not run after an indeterminate hard oracle")

    monkeypatch.setattr(cbs_fitness_module, "CognitiveBoundedSearch", _UnexpectedPCBS)

    metrics = cbs_fitness_module.compute_cbs_fitness(_simple_grid())

    assert metrics["fitness"] == -10.0
    assert metrics["oracle_status"] == "timeout"
    assert metrics["oracle_indeterminate"] is True
    assert metrics["solvable_astar"] is False
    assert math.isnan(metrics["confusion_ratio"])
