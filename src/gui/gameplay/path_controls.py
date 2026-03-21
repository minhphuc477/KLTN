"""Helpers for path reset/clear/preview orchestration."""

from typing import Any


def reset_map(gui: Any) -> None:
    """Reset current map and clear preview state."""
    gui._load_current_map()
    gui._center_view()
    if getattr(gui, "effects", None):
        gui.effects.clear()
    gui.step_count = 0
    gui.path_preview_mode = False
    gui.preview_overlay_visible = False
    gui.path_preview_dialog = None
    gui.preview_on_next_solver_result = False
    gui.message = "Map Reset"


def clear_path(gui: Any) -> None:
    """Clear active path and preview state."""
    gui.auto_path = []
    gui.solution_path = []
    gui.auto_mode = False
    gui.auto_step_idx = 0
    gui.path_preview_mode = False
    gui.preview_overlay_visible = False
    gui.path_preview_dialog = None
    gui.preview_on_next_solver_result = False
    gui.message = "Path cleared"


def show_path_preview(gui: Any, dialog_factory: Any, logger: Any) -> None:
    """Open path preview immediately or schedule it after solve completion."""
    path = list(getattr(gui, "auto_path", []) or [])
    if not path:
        path = list(getattr(gui, "solution_path", []) or [])

    if path:
        gui.auto_path = [tuple(p) for p in path]
        gui.auto_mode = False
        gui.auto_step_idx = 0
        gui.preview_on_next_solver_result = False

        try:
            gui.path_preview_dialog = dialog_factory(
                path=gui.auto_path,
                env=gui.env,
                solver_result=(getattr(gui, "solver_result", None) or {}),
                speed_multiplier=getattr(gui, "speed_multiplier", 1.0),
            )
            if getattr(gui, "preview_modal_enabled", False):
                gui.path_preview_mode = True
                gui.preview_overlay_visible = False
                gui.message = "Path preview opened (modal)"
            else:
                gui.path_preview_mode = False
                gui.preview_overlay_visible = True
                gui.message = "Path preview ready (Enter to start, Esc to dismiss)"
        except Exception as e:
            logger.exception("Failed to create path preview dialog")
            gui.path_preview_dialog = None
            gui.path_preview_mode = False
            gui.preview_overlay_visible = False
            gui.message = f"Path preview failed: {e}"
        return

    gui.preview_on_next_solver_result = True
    if getattr(gui, "solver_running", False):
        gui.message = "Solver running, preview will open when done"
        return

    gui.message = "No path yet, solving for preview..."
    gui._start_auto_solve()
