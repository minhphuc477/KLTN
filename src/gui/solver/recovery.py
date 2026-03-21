"""Helpers for solver recovery checks and timeout handling."""

from typing import Any, Callable


def log_active_solver_state(gui: Any, logger: Any, os_module: Any, time_module: Any) -> None:
    """Emit detailed state diagnostics for the active solver."""
    logger.warning("DEBUG_SOLVER: Solver state dump:")
    logger.warning("  solver_running=%s", getattr(gui, "solver_running", None))
    logger.warning("  solver_done=%s", getattr(gui, "solver_done", None))
    logger.warning(
        "  solver_proc=%s (alive=%s)",
        getattr(gui, "solver_proc", None),
        getattr(getattr(gui, "solver_proc", None), "is_alive", lambda: "N/A")(),
    )
    logger.warning(
        "  solver_outfile=%s (exists=%s)",
        getattr(gui, "solver_outfile", None),
        os_module.path.exists(getattr(gui, "solver_outfile", "") or ""),
    )
    logger.warning(
        "  solver_start_time=%s (age=%.1fs)",
        getattr(gui, "solver_start_time", None),
        (time_module.time() - getattr(gui, "solver_start_time", time_module.time())),
    )


def compute_solver_timeout_seconds(
    gui: Any,
    active_alg: int,
    default_solver_timeout_for_algorithm: Callable[[int], float],
    scale_timeout_by_grid_size: Callable[[float, int], float],
    np_module: Any,
    os_module: Any,
) -> float:
    """Compute timeout using algorithm defaults, grid-size scaling, and env override."""
    default_timeout = default_solver_timeout_for_algorithm(active_alg)
    try:
        current_map = gui.maps[gui.current_map_idx]
        grid_ref = current_map.global_grid if hasattr(current_map, "global_grid") else current_map
        grid_cells = int(np_module.asarray(grid_ref).size)
        default_timeout = scale_timeout_by_grid_size(default_timeout, grid_cells)
    except Exception:
        pass
    return float(os_module.environ.get("KLTN_SOLVER_TIMEOUT", str(default_timeout)))


def terminate_hung_solver_process(proc: Any, logger: Any) -> None:
    """Best-effort terminate then kill a hung solver process."""
    try:
        logger.warning("DEBUG_SOLVER: Terminating hung process pid=%s", getattr(proc, "pid", "N/A"))
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            logger.error("DEBUG_SOLVER: Process still alive after terminate, trying kill")
            proc.kill()
            proc.join(timeout=0.5)
    except Exception as exc:
        logger.exception("DEBUG_SOLVER: Failed to terminate hung process: %s", exc)


def force_solver_recovery_state(gui: Any, recovery_reason: str, logger: Any) -> None:
    """Force solver-related state into clean idle values."""
    logger.error("DEBUG_SOLVER: RECOVERY TRIGGERED - %s", recovery_reason)
    logger.error("DEBUG_SOLVER: Force-cleaning stuck solver state")
    gui.solver_running = False
    gui.solver_proc = None
    gui.solver_thread = None
    gui.solver_outfile = None
    gui.solver_gridfile = None
    gui.solver_start_time = None
    gui.solver_starting = False
    if hasattr(gui, "solver_algorithm_idx"):
        delattr(gui, "solver_algorithm_idx")
    gui._set_message(f"Recovered: {recovery_reason[:40]}")


def prepare_active_solver_for_new_start(
    gui: Any,
    logger: Any,
    time_module: Any,
    evaluate_solver_recovery_state: Callable[..., tuple[bool, str]],
    compute_timeout_seconds: Callable[[int], float],
    terminate_hung_process: Callable[[Any], None],
    force_recovery_state: Callable[[str], None],
    log_active_state: Callable[[], None],
) -> bool:
    """Return True when a new solver run may proceed, False to block startup."""
    if not getattr(gui, "solver_running", False):
        return True

    gui._set_message("Solver already running", 1.5)
    logger.warning("DEBUG_SOLVER: Solver already running - evaluating recovery gate")
    log_active_state()

    proc = getattr(gui, "solver_proc", None)
    done = getattr(gui, "solver_done", False)
    start_time = getattr(gui, "solver_start_time", None)
    try:
        proc_alive = proc.is_alive() if proc else False
    except Exception:
        proc_alive = False

    solver_age = (time_module.time() - start_time) if start_time else 0
    active_alg = int(getattr(gui, "solver_algorithm_idx", getattr(gui, "algorithm_idx", 0)))
    solver_timeout = compute_timeout_seconds(active_alg)

    needs_recovery, recovery_reason = evaluate_solver_recovery_state(
        has_process=bool(proc),
        process_alive=bool(proc_alive),
        solver_done=bool(done),
        solver_age=float(solver_age),
        solver_timeout=float(solver_timeout),
    )

    if needs_recovery and proc and not proc_alive and not done:
        recovery_reason = f"Process dead (exitcode={getattr(proc, 'exitcode', 'N/A')}) but not marked done"
    if needs_recovery and proc and proc_alive and solver_age > solver_timeout:
        terminate_hung_process(proc)

    if needs_recovery:
        force_recovery_state(recovery_reason)
        logger.info("DEBUG_SOLVER: Recovery complete - retrying solver start")
        return True

    logger.warning("DEBUG_SOLVER: Solver legitimately running (age=%.1fs < %.1fs timeout)", solver_age, solver_timeout)
    return False
