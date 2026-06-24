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
from src.evaluation.pcbs_telemetry_calibration import _path_efficiency


def _simple_grid() -> np.ndarray:
    grid = np.full((6, 8), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int64)
    grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[4, 6] = int(SEMANTIC_PALETTE["TRIFORCE"])
    return grid


def test_oracle_status_normalizes_common_failure_modes():
    assert oracle_status_from_outcome(True, "") == "solved"
    assert oracle_status_from_outcome(False, "Timeout: explored 100 states") == "timeout"
    assert oracle_status_from_outcome(False, "No path: all reachable states explored") == "no_path"
    assert oracle_status_from_outcome(False, "No goal (TRIFORCE) found in map") == "invalid_map"
    assert oracle_status_from_outcome(False, "misc error") == "failed"


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
