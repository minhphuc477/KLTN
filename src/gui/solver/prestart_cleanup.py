"""Helpers to cleanup preview artifacts and reset visual solver state."""

from typing import Any


def cleanup_preview_before_solver_start(gui: Any, logger: Any, os_module: Any) -> None:
    """Stop preview workers/files so new solve starts from a clean state."""
    try:
        preview_proc = getattr(gui, "preview_proc", None)
        if preview_proc and preview_proc.is_alive():
            logger.info(
                "DEBUG_SOLVER: Terminating existing preview process pid=%s",
                getattr(preview_proc, "pid", None),
            )
            try:
                preview_proc.terminate()
            except Exception:
                logger.exception("DEBUG_SOLVER: Failed to terminate preview process")
            try:
                preview_proc.join(timeout=0.2)
            except Exception:
                pass
    except Exception:
        logger.exception("DEBUG_SOLVER: Error while stopping preview process")

    try:
        out_file = getattr(gui, "preview_outfile", None)
        if out_file and os_module.path.exists(out_file):
            os_module.remove(out_file)
    except Exception:
        pass

    try:
        grid_file = getattr(gui, "preview_gridfile", None)
        if grid_file and os_module.path.exists(grid_file):
            os_module.remove(grid_file)
    except Exception:
        pass

    gui.preview_done = False
    gui.preview_proc = None
    gui.preview_outfile = None
    gui.preview_gridfile = None


def reset_solver_visual_state_before_start(gui: Any) -> None:
    """Clear solver/visual state from previous runs before scheduling a new solve."""
    gui.solver_done = False
    gui.solver_thread = None
    gui.auto_path = []
    gui.auto_mode = False
    gui.auto_step_idx = 0
    gui.block_push_animations = []
    gui.door_unlock_times = {}
    gui.collected_items = []
    gui.collected_positions = set()
