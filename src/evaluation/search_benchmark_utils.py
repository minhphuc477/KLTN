"""
Utilities for search-benchmark accounting.

These helpers keep benchmark semantics consistent across scripts:
- distinguish solved / timeout / no-path / invalid-map / failed
- avoid treating timeouts as proven unsolvable
- keep path-efficiency and confusion-ratio math aligned
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from src.simulation.search_status import oracle_status_from_outcome
from src.simulation.validator import SolverDiagnostics, StateSpaceAStar


def safe_positive_int(value: Any, default: int = 1, maximum: int = 2_147_483_647) -> int:
    """Convert telemetry/config values to a bounded positive int without crashing."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
    if not math.isfinite(numeric):
        return int(maximum)
    return int(max(1, min(float(maximum), numeric)))


def path_efficiency_ratio(path_length: int, manhattan_distance: int) -> float:
    """
    Return a bounded path-efficiency ratio in [0, 1].

    Higher is better: straight-line-optimal paths approach 1.0.
    """
    if path_length is None or not math.isfinite(float(path_length)):
        return 0.0
    if manhattan_distance is None or not math.isfinite(float(manhattan_distance)):
        return 0.0
    path_length_i = int(path_length)
    manhattan_i = int(manhattan_distance)
    if path_length_i <= 0 or manhattan_i <= 0:
        return 0.0
    return float(max(0.0, min(1.0, float(manhattan_i) / float(max(1, path_length_i)))))


def confusion_ratio_vs_oracle(
    oracle_path_length: int,
    candidate_path_length: int,
    *,
    oracle_status: str,
    candidate_success: bool,
) -> float:
    """
    Compute excess candidate path length relative to the solved oracle.

    Returns NaN when the ratio is undefined rather than polluting summaries with
    +/-inf sentinels.
    """
    if str(oracle_status) != "solved" or not bool(candidate_success):
        return float("nan")
    if oracle_path_length is None or not math.isfinite(float(oracle_path_length)):
        return float("inf")
    if candidate_path_length is None or not math.isfinite(float(candidate_path_length)):
        return float("inf")
    oracle_len = int(oracle_path_length)
    candidate_len = int(candidate_path_length)
    if oracle_len < 0 or candidate_len < 0:
        return float("nan")
    if oracle_len == 0:
        return 0.0 if candidate_len == 0 else float(max(0, candidate_len))
    return float(max(0, candidate_len - oracle_len)) / float(oracle_len)


def normalized_confusion_ratio(
    oracle_path_length: int,
    candidate_path_length: int,
    manhattan_distance: int = 0,
    *,
    oracle_status: str = "solved",
    candidate_success: bool = True,
) -> float:
    """
    Return excess-path confusion normalized by a robust lower bound.

    This avoids treating a 2-step overhead on a tiny dungeon as equivalent to a
    200-step overhead on a large dungeon. Returns NaN when the oracle/candidate
    comparison is undefined.
    """
    if str(oracle_status) != "solved" or not bool(candidate_success):
        return float("nan")
    if oracle_path_length is None or not math.isfinite(float(oracle_path_length)):
        return float("inf")
    if candidate_path_length is None or not math.isfinite(float(candidate_path_length)):
        return float("inf")
    if manhattan_distance is None or not math.isfinite(float(manhattan_distance)):
        return float("inf")
    oracle_len = int(oracle_path_length)
    candidate_len = int(candidate_path_length)
    manhattan_i = int(manhattan_distance)
    if oracle_len < 0 or candidate_len < 0:
        return float("nan")
    denominator = max(1, oracle_len, manhattan_i)
    ratio = float(max(0, candidate_len - oracle_len)) / float(denominator)
    return float(max(0.0, min(1.0, ratio)))


def finite_mean(values: List[Any]) -> float:
    """Average only finite numeric values; return 0.0 when empty."""
    finite: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    if not finite:
        return 0.0
    return float(sum(finite) / len(finite))


def run_astar_oracle(env: Any, timeout: int, heuristic_mode: str = "balanced") -> Dict[str, Any]:
    """Run canonical A* with diagnostics and return a normalized payload."""
    priority_options = {
        "allow_diagonals": False,
        "rules_profile": "vglc_strict",
        "representation": "tile",
        "enable_hierarchical": False,
    }
    solver = StateSpaceAStar(
        env,
        timeout=safe_positive_int(timeout),
        heuristic_mode=str(heuristic_mode or "balanced"),
        priority_options=priority_options,
        search_mode="astar",
    )
    success, path, diagnostics = solver.solve_with_diagnostics()
    if not isinstance(diagnostics, SolverDiagnostics):
        diagnostics = SolverDiagnostics(success=bool(success), states_explored=0)
    solver_used = "astar"
    primary_failure = str(diagnostics.failure_reason or "")

    if not bool(success):
        env.reset()
        fallback = StateSpaceAStar(
            env,
            timeout=safe_positive_int(timeout),
            heuristic_mode=str(heuristic_mode or "balanced"),
            priority_options=priority_options,
            search_mode="dijkstra",
        )
        fb_success, fb_path, fb_diag = fallback.solve_with_diagnostics()
        if isinstance(fb_diag, SolverDiagnostics) and bool(fb_success):
            success = True
            path = fb_path
            diagnostics = fb_diag
            solver_used = "dijkstra_fallback"

    status = oracle_status_from_outcome(bool(success), diagnostics.failure_reason)
    path_list: List[Tuple[int, int]] = list(path or [])
    return {
        "success": bool(success),
        "path": path_list,
        "path_length": int(max(0, len(path_list) - 1)),
        "states_explored": int(diagnostics.states_explored or 0),
        "status": status,
        "failure_reason": str(diagnostics.failure_reason or ""),
        "time_ms": float(diagnostics.time_taken_ms or 0.0),
        "states_pruned_dominated": int(diagnostics.states_pruned_dominated or 0),
        "max_queue_size": int(diagnostics.max_queue_size or 0),
        "timeout_flag": int(status == "timeout"),
        "final_inventory": dict(diagnostics.final_inventory or {}),
        "solver_used": str(solver_used),
        "primary_solver_error": primary_failure,
    }
