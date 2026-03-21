"""Core helpers for solver state cleanup and dropdown synchronization."""

from typing import Any, Callable, Iterable, Tuple


def clear_solver_state(gui: Any, reason: str, logger: Any) -> None:
    """Centralize solver state cleanup and ensure consistency."""
    logger.info("SOLVER_CLEANUP: Clearing solver state (%s)", reason)
    gui.solver_running = False
    gui.solver_done = True
    gui.solver_proc = None
    gui.solver_thread = None
    gui.solver_outfile = None
    gui.solver_gridfile = None
    gui.solver_start_time = None
    gui.solver_starting = False
    if hasattr(gui, "solver_algorithm_idx"):
        delattr(gui, "solver_algorithm_idx")
    logger.debug("SOLVER_CLEANUP: State cleared")


def sync_solver_dropdown_settings(
    gui: Any,
    sync_fn: Callable[[int, str, float, Iterable], Tuple[int, str, float]],
) -> Tuple[int, str, float]:
    """Refresh algorithm/representation/ARA values from dropdown widgets."""
    widgets = gui.widget_manager.widgets if hasattr(gui, "widget_manager") and gui.widget_manager else []
    alg_idx, rep_mode, ara_weight = sync_fn(
        getattr(gui, "algorithm_idx", 0),
        getattr(gui, "search_representation", "hybrid"),
        getattr(gui, "ara_weight", 1.0),
        widgets,
    )
    gui.algorithm_idx = alg_idx
    gui.search_representation = rep_mode
    gui.ara_weight = float(ara_weight)
    return alg_idx, rep_mode, float(ara_weight)
