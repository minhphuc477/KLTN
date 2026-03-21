"""Helpers to bootstrap solver worker launch with test mode and fallback."""

import os
from typing import Any, Callable


def launch_solver_worker(
    gui: Any,
    kwargs: dict,
    logger: Any,
    launch_solver_process: Callable[..., None],
    start_solver_thread_fallback: Callable[..., None],
    multiprocessing_module: Any,
) -> None:
    """Launch solver process, with thread-based fallback on process failure."""
    try:
        if os.environ.get("KLTN_SOLVER_TEST") == "1":
            import time as _time

            proc = multiprocessing_module.Process(target=_time.sleep, args=(2,), daemon=True)
            proc.start()
            gui.solver_proc = proc
            gui.solver_thread = None
            gui.solver_starting = False
            gui._set_message("Test solver process started (sleep)")
            return

        launch_solver_process(**kwargs)
    except Exception as exc:
        logger.exception("SOLVER: Failed to start solver process: %s", exc)
        logger.info("SOLVER: Falling back to thread-based solver")
        gui.solver_starting = False
        start_solver_thread_fallback(**kwargs)
