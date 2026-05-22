"""Helpers for route payload shaping and application."""

import os
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple

from src.gui.solver.pcbs_route import compress_pcbs_route_for_replay


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

    loaded_path = [tuple(p) for p in route_data["path"]]
    algorithm_label = str(route_data.get("algorithm", "") or "")
    solver_result = dict(route_data.get("solver_result") or {})
    should_compress_pcbs = (
        "P-CBS" in algorithm_label.upper()
        or str(solver_result.get("algorithm", "")).strip().upper() == "P-CBS"
        or bool(solver_result.get("cbs_metrics"))
    )
    route_mode = str(os.environ.get("KLTN_PCBS_ROUTE_MODE", "solution") or "solution").strip().lower()
    if should_compress_pcbs and route_mode not in {"trace", "trajectory", "raw"} and getattr(gui, "env", None) is not None:
        compressed, stats = compress_pcbs_route_for_replay(
            grid=getattr(gui.env, "original_grid", None),
            path=loaded_path,
            solver_options=getattr(gui.env, "solver_options", None),
        )
        if not stats.get("compression_error"):
            loaded_path = compressed
            solver_result.update(
                {
                    "trajectory_len": int(stats.get("raw_trajectory_len", len(route_data["path"])) or len(route_data["path"])),
                    "display_path_len": int(stats.get("display_path_len", len(loaded_path)) or len(loaded_path)),
                    "pcbs_route_mode": route_mode,
                    "pcbs_route_compressed": bool(stats.get("compressed", False)),
                    "pcbs_loops_removed": int(stats.get("loops_removed", 0) or 0),
                }
            )

    gui.solution_path = loaded_path
    gui.auto_path = list(gui.solution_path)
    gui.auto_step_idx = 0
    gui.auto_mode = False

    if "algorithm" in route_data:
        gui.last_algorithm = route_data["algorithm"]
    if "solve_time_ms" in route_data:
        gui.last_solve_time = route_data["solve_time_ms"] / 1000.0
    if "nodes_explored" in route_data:
        gui.last_nodes_explored = route_data["nodes_explored"]
    if "solver_result" in route_data or solver_result:
        gui.solver_result = solver_result
    gui.loaded_route_source = route_data.get("source_solver_case_study") or route_data.get("source_artifact") or "route_json"

    return len(gui.solution_path)
