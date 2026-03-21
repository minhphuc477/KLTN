"""Helpers for synchronous solver execution in debug mode."""

from typing import Any, Callable, Optional


def run_solver_sync(gui: Any, logger: Any, solve_in_subprocess: Callable[..., dict], algorithm_idx: Optional[int] = None) -> None:
    """Run solver synchronously in main thread to bypass multiprocessing issues."""
    logger.warning("DEBUG_SYNC: Starting synchronous solver (UI will freeze)")
    gui._set_message("Running solver synchronously (debug)...", 5.0)
    gui._sync_solver_dropdown_settings()

    request = gui._build_solver_request(algorithm_idx=algorithm_idx)
    if request is None:
        logger.error("DEBUG_SYNC: No start/goal defined")
        return

    grid_arr = request["grid_arr"]
    graph = request["graph"]
    room_to_node = request["room_to_node"]
    room_positions = request["room_positions"]
    node_to_room = request["node_to_room"]
    start = request["start"]
    goal = request["goal"]
    alg_idx = request["alg_idx"]
    flags = request["flags"]
    priority_options = request["priority_options"]

    logger.info("DEBUG_SYNC: Calling _solve_in_subprocess with start=%s, goal=%s", start, goal)

    try:
        result = solve_in_subprocess(
            grid_arr,
            start,
            goal,
            alg_idx,
            flags,
            priority_options,
            graph=graph,
            room_to_node=room_to_node,
            room_positions=room_positions,
            node_to_room=node_to_room,
        )

        logger.info(
            "DEBUG_SYNC: Solver returned: success=%s, path_len=%d",
            result.get("success"),
            len(result.get("path", []) or []),
        )

        if result.get("success") and result.get("path"):
            gui.auto_path = result["path"]
            solver_result = result.get("solver_result", {})

            logger.info(
                "DEBUG_SYNC: Path loaded successfully, first=%s, last=%s",
                gui.auto_path[0] if gui.auto_path else None,
                gui.auto_path[-1] if gui.auto_path else None,
            )

            logger.info("DEBUG_SYNC: Calling _execute_auto_solve()")
            gui._execute_auto_solve(gui.auto_path, solver_result, teleports=0)
            gui._set_message(f"DEBUG: Solver done! Path: {len(gui.auto_path)} steps. auto_mode={gui.auto_mode}")

            logger.info("DEBUG_SYNC: After execute: auto_mode=%s, auto_step_idx=%s", gui.auto_mode, gui.auto_step_idx)
        else:
            msg = result.get("message") or "No path found"
            logger.warning("DEBUG_SYNC: Solver failed: %s", msg)
            gui._set_message(f"DEBUG: No path - {msg}")

    except Exception as exc:
        logger.exception("DEBUG_SYNC: Solver exception")
        gui._set_message(f"DEBUG: Solver error - {exc}")
