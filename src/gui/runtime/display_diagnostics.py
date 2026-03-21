"""Display diagnostics helpers for GUI runtime state and watchdog actions."""

from typing import Any


def handle_watchdog_screenshot(gui: Any, pygame: Any, logger: Any, os_module: Any) -> bool:
    """Save requested watchdog screenshot and always clear request flag."""
    shot = getattr(gui, "_watchdog_request_screenshot", None)
    if not shot:
        return False
    try:
        surf = pygame.display.get_surface()
        if not surf:
            logger.warning("Watchdog requested screenshot but no display surface available")
            try:
                gui._watchdog_request_screenshot = None
            except Exception:
                pass
            return False
        try:
            pygame.image.save(surf, shot)
            logger.warning("Watchdog screenshot saved by main thread: %s", shot)
            gui._show_toast(f"Watchdog screenshot: {os_module.path.basename(shot)}", 3.0, "info")
            return True
        except Exception:
            logger.exception("Failed to save watchdog screenshot on main thread")
            return False
    finally:
        try:
            gui._watchdog_request_screenshot = None
        except Exception:
            pass


def report_ui_state(gui: Any, logger: Any) -> dict:
    """Build a diagnostic dictionary describing current GUI/UI runtime state."""
    try:
        return {
            "fullscreen": gui.fullscreen,
            "screen_w": getattr(gui, "screen_w", None),
            "screen_h": getattr(gui, "screen_h", None),
            "preview_overlay_visible": getattr(gui, "preview_overlay_visible", False),
            "preview_modal_enabled": getattr(gui, "preview_modal_enabled", False),
            "control_panel_enabled": getattr(gui, "control_panel_enabled", False),
            "control_panel_rect": getattr(gui, "control_panel_rect", None),
            "control_panel_collapsed": getattr(gui, "control_panel_collapsed", False),
            "solver_running": getattr(gui, "solver_running", False),
            "solver_proc_alive": getattr(getattr(gui, "solver_proc", None), "is_alive", lambda: False)(),
            "solver_comparison_thread_alive": getattr(
                getattr(gui, "solver_comparison_thread", None),
                "is_alive",
                lambda: False,
            )(),
            "debug_click_log_len": len(getattr(gui, "debug_click_log", [])),
        }
    except Exception:
        logger.exception("Failed to build UI state report")
        return {}
