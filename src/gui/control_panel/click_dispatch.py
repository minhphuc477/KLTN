"""Control-panel click dispatch orchestration for ZeldaGUI wrappers."""

from __future__ import annotations


def handle_control_panel_click(gui, pos, button, event_type, *, logger, debug_input_active):
    """Dispatch click events for control-panel widgets and scrolling behavior."""
    if not gui.control_panel_enabled or not gui.widget_manager:
        return False

    if event_type == "down":
        panel_hit_rect = gui._control_panel_hit_rect()
        if gui._should_swallow_control_panel_click(panel_hit_rect, pos):
            return True
        sc_pos = gui._translate_control_panel_click(pos, panel_hit_rect)

        outside_result = gui._handle_outside_control_panel_click(panel_hit_rect, pos, button)
        if outside_result is not None:
            return outside_result

        logger.debug(
            "Control panel click: pos=%s sc_pos=%s scroll=%s header_h=%s",
            pos,
            sc_pos,
            getattr(gui, "control_panel_scroll", 0),
            45,
        )

        any_contains = gui._refresh_control_panel_layout_if_needed(sc_pos)

        handled = gui.widget_manager.handle_mouse_down(sc_pos, button)

        logger.debug(
            "Control panel click handled=%s at pos=%s sc_pos=%s any_contains=%s",
            handled,
            pos,
            sc_pos,
            any_contains,
        )
        if not handled:
            if debug_input_active:
                try:
                    gui._dump_control_panel_widget_state(pos)
                except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                    logger.exception("Failed to dump widget hit tests after unhandled click")
            handled = gui._retry_control_panel_click_after_auto_scroll(pos, sc_pos, button, handled)

        if handled:
            logger.debug("Control panel click handled by widget manager at pos=%r (button=%r)", pos, button)
            gui._apply_control_panel_widget_updates()

        return handled

    if event_type == "up":
        if (
            getattr(gui, "control_panel_can_scroll", False)
            and getattr(gui, "control_panel_rect", None)
            and gui.control_panel_rect.collidepoint(pos)
        ):
            sc_pos = (pos[0], pos[1] + getattr(gui, "control_panel_scroll", 0))
        else:
            sc_pos = pos
        return gui.widget_manager.handle_mouse_up(sc_pos, button)

    return False

