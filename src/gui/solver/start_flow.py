"""Helpers for orchestrating solver startup flow from GUI state."""

from typing import Any


def start_auto_solve(gui: Any, logger: Any, debug_sync_solver: bool) -> None:
    """Start auto-solve mode using state-space solver with inventory tracking."""
    alg_idx, rep_mode, ara_weight = gui._sync_solver_dropdown_settings()
    debug_solver_flow = bool(getattr(gui, "debug_solver_flow", False))
    logger.info(
        "SOLVER_FIX: Synced settings from dropdowns -> alg_idx=%d, representation=%s, ara_weight=%.2f",
        alg_idx,
        rep_mode,
        ara_weight,
    )

    alg_name = gui._algorithm_name(alg_idx)
    if debug_solver_flow:
        logger.info("SOLVER_FLOW: _start_auto_solve() called")
        logger.info("SOLVER_FLOW: Algorithm: %s (idx=%d)", alg_name, alg_idx)
        logger.info(
            "SOLVER_FLOW: Search representation: %s, ara_weight=%.2f",
            gui.search_representation,
            gui.ara_weight,
        )
        logger.info(
            "SOLVER_FLOW: solver_running=%s, auto_mode=%s, auto_start_solver=%s",
            getattr(gui, "solver_running", None),
            getattr(gui, "auto_mode", None),
            getattr(gui, "auto_start_solver", None),
        )

    if (
        getattr(gui, "auto_path", None)
        and not getattr(gui, "auto_mode", False)
        and getattr(gui, "loaded_route_source", None)
        and getattr(gui, "use_preloaded_route_on_solve", False)
    ):
        logger.info(
            "SOLVER: Starting preloaded route instead of recomputing solver path (source=%s, len=%d)",
            getattr(gui, "loaded_route_source", None),
            len(getattr(gui, "auto_path", []) or []),
        )
        gui._execute_auto_solve_from_preview()
        return

    if not gui._prepare_active_solver_for_new_start():
        return

    precheck_ok, precheck_msg = gui._run_prechecks_and_optional_prune()
    if not precheck_ok:
        gui._set_message(precheck_msg or "Precheck failed", 4.0)
        logger.warning("PRECHECK: Solve blocked: %s", precheck_msg)
        return
    if precheck_msg:
        logger.info("PRECHECK: %s", precheck_msg)

    gui._cleanup_preview_before_solver_start()
    gui._reset_solver_visual_state_before_start()
    if debug_solver_flow:
        logger.info("SOLVER_FLOW: Cleared previous state, solver_done=False")

    if debug_sync_solver:
        logger.warning("SYNC_SOLVER: Running solver synchronously (blocking)")
        gui._run_solver_sync(algorithm_idx=alg_idx)
        return

    gui._set_message("Starting solver in background...", 2.0)
    try:
        gui._schedule_solver(algorithm_idx=alg_idx)
        if debug_solver_flow:
            logger.info("SOLVER_FLOW: _schedule_solver() completed without exception")
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.exception("Failed to schedule solver")
        gui._set_message("Failed to start solver", 3.0)
        gui.preview_on_next_solver_result = False

