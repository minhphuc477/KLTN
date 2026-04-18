from __future__ import annotations

import math

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.evaluation.search_benchmark_utils import (
    confusion_ratio_vs_oracle,
    finite_mean,
    oracle_status_from_outcome,
    path_efficiency_ratio,
    run_astar_oracle,
)
from src.simulation.validator import ZeldaLogicEnv


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


def test_confusion_ratio_vs_oracle_returns_nan_when_oracle_not_resolved():
    assert math.isnan(confusion_ratio_vs_oracle(12, 20, oracle_status="timeout", candidate_success=True))
    assert math.isnan(confusion_ratio_vs_oracle(12, 20, oracle_status="solved", candidate_success=False))
    assert confusion_ratio_vs_oracle(10, 15, oracle_status="solved", candidate_success=True) == 1.5


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
