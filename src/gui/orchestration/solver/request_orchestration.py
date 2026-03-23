"""Orchestration helpers for building solver requests from GUI state."""

from typing import Any, Optional

from src.gui.solver.request_helpers import (
    build_solver_request as _build_solver_request,
    get_solver_map_context as _get_solver_map_context,
)


def get_solver_map_context(
    gui: Any,
    get_solver_map_context_helper: Any = None,
) -> dict:
    """Return current grid and optional topology context needed by solver backends."""
    current = gui.maps[gui.current_map_idx]
    helper = get_solver_map_context_helper or _get_solver_map_context
    return helper(current)


def build_solver_request(
    gui: Any,
    build_solver_request_helper: Any = None,
    algorithm_idx: Optional[int] = None,
    on_missing_message: str = "Start/goal not defined for this map",
) -> Optional[dict]:
    """Build canonical solver request payload from current GUI state."""
    alg_idx = algorithm_idx if algorithm_idx is not None else getattr(gui, "algorithm_idx", 0)
    helper = build_solver_request_helper or _build_solver_request
    request = helper(
        current_map=gui.maps[gui.current_map_idx],
        env=gui.env,
        feature_flags=gui.feature_flags,
        algorithm_idx=alg_idx,
        ara_weight=float(getattr(gui, "ara_weight", 1.0)),
        search_representation=str(getattr(gui, "search_representation", "hybrid")),
    )
    if request is None and on_missing_message:
        gui._set_message(on_missing_message)
    return request
