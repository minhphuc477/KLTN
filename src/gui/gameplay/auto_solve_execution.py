"""Helpers for auto-solve execution state transitions and messaging."""

from typing import Any


def _reset_visual_run_state(gui: Any) -> None:
    """Reset animation and item/inventory state for a new visual run."""
    gui.auto_step_idx = 0
    gui.auto_mode = True
    gui.auto_step_timer = 0.0
    gui._auto_stuck_retries = 0
    gui.preview_on_next_solver_result = False

    gui.keys_used = 0
    gui.bombs_used = 0
    gui.boss_keys_used = 0
    gui.used_items = []

    gui.collected_items = []
    gui.collected_positions = set()
    gui.keys_collected = 0
    gui.bombs_collected = 0
    gui.boss_keys_collected = 0
    gui.item_type_map = {}
    gui.item_pickup_times = {}


def execute_auto_solve(gui: Any, path: Any, solver_result: Any, teleports: int, logger: Any) -> None:
    """Execute auto-solve immediately without preview dialog."""
    if not path or len(path) == 0:
        logger.error("EXECUTE: Refusing to start animation with empty path")
        gui._show_error("No valid path to animate")
        return

    logger.info("EXECUTE: path=%d steps, setting auto_mode=True", len(path) if path else 0)
    logger.info(
        "EXECUTE: Before state: auto_mode=%s, auto_step_idx=%s",
        getattr(gui, "auto_mode", None),
        getattr(gui, "auto_step_idx", None),
    )

    gui.auto_path = [tuple(p) for p in path]
    _reset_visual_run_state(gui)

    if gui.env is None:
        logger.error("EXECUTE: environment is not initialized")
        gui.auto_mode = False
        gui._show_error("Environment not initialized")
        return
    gui.env.reset()

    if solver_result and "cbs_metrics" in solver_result:
        gui.last_solver_metrics = {
            "name": f"CBS ({solver_result.get('persona', 'unknown')})",
            "nodes": solver_result.get("nodes", 0),
            "path_len": len(path),
            "cbs": solver_result["cbs_metrics"],
        }

    try:
        gui._scan_items_along_path(path)
        items_text = gui._get_path_items_display_text()
        if items_text:
            logger.info("EXECUTE: Path items preview: %s", items_text)
    except Exception as scan_err:
        logger.warning("EXECUTE: Failed to scan path items: %s", scan_err)

    logger.info(
        "EXECUTE: After state: auto_mode=%s, auto_step_idx=%s, auto_path_len=%d",
        gui.auto_mode,
        gui.auto_step_idx,
        len(gui.auto_path) if gui.auto_path else 0,
    )

    keys_used = solver_result.get("keys_used", 0) if solver_result else 0
    keys_avail = solver_result.get("keys_available", 0) if solver_result else 0
    key_info = f"Keys: {keys_avail}->{keys_avail - keys_used}" if keys_used > 0 else ""
    items_display = gui._get_path_items_display_text()

    if solver_result and "cbs_metrics" in solver_result:
        cbs = solver_result["cbs_metrics"]
        persona = solver_result.get("persona", "unknown")
        base_msg = f"CBS ({persona.title()}): {len(path)} steps"
        metrics_msg = f"Confusion: {cbs['confusion_index']:.2f} | Cognitive Load: {cbs['cognitive_load']:.2f}"
        gui.message = f"{base_msg} | {metrics_msg}"
        toast_msg = (
            f"CBS ({persona.title()}) completed | "
            f"Confusion: {cbs['confusion_index']:.2f} | Entropy: {cbs['navigation_entropy']:.2f}"
        )
        gui._show_toast(toast_msg, duration=4.0, toast_type="success")
    elif teleports > 0:
        gui.message = f"Path: {len(path)} ({teleports} warps) {key_info}"
    else:
        base_msg = f"Path: {len(path)} steps"
        if items_display:
            gui.message = f"{base_msg} | Items: {items_display}"
        elif key_info:
            gui.message = f"{base_msg} {key_info}"
        else:
            gui.message = base_msg


def execute_auto_solve_from_preview(gui: Any, logger: Any) -> None:
    """Start auto-solve after preview confirmation."""
    if not getattr(gui, "auto_path", None):
        gui.auto_mode = False
        gui._show_error("No preview path available")
        return

    gui.auto_path = [tuple(p) for p in gui.auto_path]
    _reset_visual_run_state(gui)

    if gui.env is None:
        gui.auto_mode = False
        gui._show_error("Environment not initialized")
        return
    gui.env.reset()

    try:
        gui._scan_items_along_path(gui.auto_path)
        items_text = gui._get_path_items_display_text()
    except Exception as scan_err:
        logger.warning("EXECUTE_PREVIEW: Failed to scan path items: %s", scan_err)
        items_text = ""

    gui.path_preview_mode = False
    preview_dialog = gui.path_preview_dialog
    gui.path_preview_dialog = None
    gui.preview_overlay_visible = False

    if preview_dialog and getattr(preview_dialog, "solver_result", None):
        gui.solver_result = preview_dialog.solver_result

    if preview_dialog:
        base_msg = f"Auto-solve started! Path: {len(gui.auto_path)} steps"
        if items_text:
            gui.message = f"{base_msg} | Items: {items_text}"
        else:
            gui.message = base_msg
    else:
        gui.message = "Auto-solve started!"
