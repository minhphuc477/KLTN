"""Pure helpers for control-panel click routing and scroll behavior."""

from time import time
from typing import Any, Callable, Optional, Tuple


def control_panel_hit_rect(
    panel_rect: Any,
    debug_control_panel: bool,
    debug_panel_click_padding: int,
    rect_factory: Optional[Callable[[int, int, int, int], Any]] = None,
) -> Any:
    """Return panel hit rect, optionally expanded for debug click padding."""
    if not panel_rect:
        return panel_rect
    padding = int(debug_panel_click_padding or 0) if debug_control_panel else 0
    if padding <= 0:
        return panel_rect
    factory = rect_factory if rect_factory is not None else type(panel_rect)
    return factory(
        panel_rect.x - padding,
        panel_rect.y,
        panel_rect.width + padding,
        panel_rect.height,
    )


def should_swallow_control_panel_click(
    dragging: bool,
    ignore_click_until: float,
    panel_hit_rect: Any,
    pos: Tuple[int, int],
    logger: Any = None,
) -> bool:
    """Return True when control panel must swallow click due to active drag window."""
    if not (bool(dragging) or time() < float(ignore_click_until or 0.0)):
        return False
    if panel_hit_rect and panel_hit_rect.collidepoint(pos):
        if logger is not None:
            logger.debug(
                "Ignored click on control panel due to active scroll/ignore window (dragging=%s ignore_until=%s) and pos inside panel_hit_rect",
                dragging,
                ignore_click_until,
            )
        return True
    return False


def translate_control_panel_click(
    pos: Tuple[int, int],
    panel_hit_rect: Any,
    panel_rect: Any,
    can_scroll: bool,
    control_panel_scroll: int,
    header_height: int = 45,
) -> Tuple[int, int]:
    """Translate click coordinates for scrolled panel content area."""
    if not (can_scroll and panel_hit_rect and panel_hit_rect.collidepoint(pos)):
        return pos
    panel_top = panel_rect.y if panel_rect is not None else 0
    if pos[1] > panel_top + int(header_height):
        return (pos[0], pos[1] + int(control_panel_scroll or 0))
    return pos


def handle_outside_control_panel_click(
    panel_hit_rect: Any,
    pos: Tuple[int, int],
    button: int,
    widget_manager: Any,
    dropdown_type: Any,
    logger: Any = None,
) -> Optional[bool]:
    """Handle clicks outside panel while a dropdown is open.

    Returns:
    - None if caller should continue normal dispatch.
    - bool when the click has been handled/decided.
    """
    try:
        if panel_hit_rect and panel_hit_rect.collidepoint(pos):
            return None
        if any(
            isinstance(w, dropdown_type) and getattr(w, "is_open", False)
            for w in widget_manager.widgets
        ):
            return widget_manager.handle_mouse_down(pos, button)
        return False
    except Exception:
        if logger is not None:
            logger.exception("Error while checking outside-panel click handling")
        return None


def refresh_control_panel_layout_if_needed(
    widget_manager: Any,
    sc_pos: Tuple[int, int],
    debug_input_active: bool,
    panel_rect: Any,
    reposition_widgets: Optional[Callable[[int, int], None]],
    logger: Any = None,
) -> bool:
    """Refresh widget rects when no widget claims click coordinates."""
    try:
        any_contains = False
        for widget in widget_manager.widgets:
            full_rect = getattr(widget, "full_rect", getattr(widget, "rect", None))
            rect = getattr(widget, "rect", None)
            if (full_rect and full_rect.collidepoint(sc_pos)) or (rect and rect.collidepoint(sc_pos)):
                any_contains = True
                break

        if any_contains:
            return True

        if logger is not None:
            if debug_input_active:
                logger.info(
                    "No widget claims sc_pos=%s: attempting _reposition_widgets to refresh layout",
                    sc_pos,
                )
            else:
                logger.debug(
                    "No widget claims sc_pos=%s: attempting _reposition_widgets to refresh layout",
                    sc_pos,
                )

        if panel_rect and reposition_widgets is not None:
            try:
                reposition_widgets(panel_rect.x, panel_rect.y)
            except Exception:
                if logger is not None:
                    logger.exception("Reposition attempt failed")
            else:
                for widget in widget_manager.widgets:
                    full_rect = getattr(widget, "full_rect", getattr(widget, "rect", None))
                    rect = getattr(widget, "rect", None)
                    if (full_rect and full_rect.collidepoint(sc_pos)) or (rect and rect.collidepoint(sc_pos)):
                        return True
        return False
    except Exception:
        if logger is not None:
            logger.exception("Failure while checking widget rects before dispatch")
        return False


def retry_control_panel_click_after_auto_scroll(
    pos: Tuple[int, int],
    sc_pos: Tuple[int, int],
    button: int,
    handled: bool,
    panel_rect: Any,
    widget_manager: Any,
    can_scroll: bool,
    control_panel_scroll: int,
    control_panel_scroll_max: int,
    logger: Any = None,
    header_height: int = 45,
    ignore_click_window_sec: float = 0.12,
) -> Tuple[bool, int, float]:
    """Auto-scroll panel to nearest clipped widget and re-dispatch click.

    Returns a tuple: (handled, new_scroll, ignore_until).
    """
    try:
        if not (panel_rect and widget_manager and can_scroll):
            return handled, int(control_panel_scroll or 0), 0.0

        click_y_local = sc_pos[1] - panel_rect.y - int(header_height)
        nearest = None
        nearest_dist = None
        for widget in widget_manager.widgets:
            full_rect = getattr(widget, "full_rect", getattr(widget, "rect", None))
            if full_rect is None:
                continue
            widget_top_rel = full_rect.y - panel_rect.y
            widget_bottom_rel = full_rect.bottom - panel_rect.y
            if widget_top_rel < int(header_height) or widget_bottom_rel > panel_rect.height:
                center = (widget_top_rel + widget_bottom_rel) / 2.0
                dist = abs(center - click_y_local)
                if nearest is None or dist < nearest_dist:
                    nearest = widget
                    nearest_dist = dist

        if nearest is None:
            return handled, int(control_panel_scroll or 0), 0.0

        prev_scroll = int(control_panel_scroll or 0)
        widget_rect = getattr(nearest, "full_rect", nearest.rect)
        target_scroll = max(
            0,
            min(
                int(control_panel_scroll_max or 0),
                widget_rect.y - panel_rect.y - int(header_height),
            ),
        )
        if abs(target_scroll - prev_scroll) <= 1:
            return handled, prev_scroll, 0.0

        ignore_until = time() + float(ignore_click_window_sec)
        if logger is not None:
            logger.info("Control panel auto-scrolled to reveal widget (scroll=%s)", target_scroll)

        new_sc_pos = (pos[0], pos[1] + target_scroll)
        try:
            handled = widget_manager.handle_mouse_down(new_sc_pos, button)
            if logger is not None:
                logger.debug("After auto-scroll, re-dispatch handled=%s", handled)
        except Exception:
            if logger is not None:
                logger.exception("Re-dispatch after auto-scroll failed")
        return handled, target_scroll, ignore_until
    except Exception:
        if logger is not None:
            logger.exception("Auto-scroll retry failed")
        return handled, int(control_panel_scroll or 0), 0.0