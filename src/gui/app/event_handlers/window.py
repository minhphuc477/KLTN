"""Window/system event handlers for GUI event loop."""

from __future__ import annotations


def handle_videoresize_event(gui, event, pygame_module, logger):
    """Handle VIDEORESIZE and immediate display/layout refresh."""
    gui.screen_w = max(event.w, gui.MIN_WIDTH)
    gui.screen_h = max(event.h, gui.MIN_HEIGHT)
    if not gui.fullscreen:
        screen = gui._safe_set_mode((gui.screen_w, gui.screen_h), pygame_module.RESIZABLE)
        if not screen:
            logger.warning("VIDEORESIZE: _safe_set_mode failed; attempting display reinit")
            try:
                gui._attempt_display_reinit()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                logger.exception("VIDEORESIZE: display reinit failed")
        else:
            gui.screen = screen
            try:
                gui.screen_w, gui.screen_h = gui.screen.get_size()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass

        try:
            gui._load_assets()
            gui._render()
            try:
                pygame_module.display.flip()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                logger.exception("Flip failed after VIDEORESIZE")
        except (AttributeError, RuntimeError, ValueError, TypeError):
            logger.exception("Failed to refresh UI after VIDEORESIZE")

    if gui.control_panel_enabled:
        gui._update_control_panel_positions()
