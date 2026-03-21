"""Helpers for route payload shaping and application."""

from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


def build_route_export_payload(gui: Any, path: Iterable[Tuple[int, int]]) -> Dict[str, Any]:
    """Build serializable route export payload from GUI state."""
    route = list(path)
    return {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "start": getattr(gui, "start_pos", None),
        "goal": getattr(gui, "goal_pos", None),
        "path": route,
        "path_length": len(route),
        "algorithm": getattr(gui, "last_algorithm", "unknown"),
        "solve_time_ms": getattr(gui, "last_solve_time", 0) * 1000,
        "nodes_explored": getattr(gui, "last_nodes_explored", 0),
    }


def apply_loaded_route_data(gui: Any, route_data: Dict[str, Any]) -> int:
    """Apply loaded route data to GUI state; returns path length."""
    gui.start_pos = tuple(route_data["start"])
    gui.goal_pos = tuple(route_data["goal"])
    gui.solution_path = [tuple(p) for p in route_data["path"]]
    gui.auto_path = list(gui.solution_path)
    gui.auto_step_idx = 0
    gui.auto_mode = False

    if "algorithm" in route_data:
        gui.last_algorithm = route_data["algorithm"]
    if "solve_time_ms" in route_data:
        gui.last_solve_time = route_data["solve_time_ms"] / 1000.0
    if "nodes_explored" in route_data:
        gui.last_nodes_explored = route_data["nodes_explored"]

    return len(gui.solution_path)
