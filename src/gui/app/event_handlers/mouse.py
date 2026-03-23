"""Mouse event handlers for GUI event loop."""

from __future__ import annotations


def handle_mouse_button_down_preamble(
    gui,
    event,
    pygame_module,
    time_module,
    logger,
    debug_input_active=False,
):
    """Run MOUSEBUTTONDOWN diagnostics and focus recovery. Returns (mouse_pos, consumed)."""
    mouse_pos = getattr(event, "pos", pygame_module.mouse.get_pos())

    try:
        focused = pygame_module.mouse.get_focused()
        grabbed = pygame_module.event.get_grab()
        pressed = pygame_module.mouse.get_pressed()
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        focused = False
        grabbed = False
        pressed = None

    logger.debug(
        "MOUSEBUTTONDOWN at %s (button=%s) fullscreen=%s focused=%s grabbed=%s pressed=%s",
        mouse_pos,
        getattr(event, "button", None),
        gui.fullscreen,
        focused,
        grabbed,
        pressed,
    )

    try:
        in_sidebar = mouse_pos[0] >= (gui.screen_w - gui.SIDEBAR_WIDTH)
        in_control_panel = bool(
            getattr(gui, "control_panel_rect", None) and gui.control_panel_rect.collidepoint(mouse_pos)
        )
        sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
        diag = {
            "preview_overlay_visible": getattr(gui, "preview_overlay_visible", False),
            "preview_modal_enabled": getattr(gui, "preview_modal_enabled", False),
            "show_solver_comparison_overlay": getattr(gui, "show_solver_comparison_overlay", False),
            "control_panel_active": getattr(gui, "control_panel_enabled", False),
            "in_sidebar": in_sidebar,
            "in_control_panel": in_control_panel,
            "control_panel_ignore_until": getattr(gui, "control_panel_ignore_click_until", 0.0),
            "control_panel_rect": (
                None
                if getattr(gui, "control_panel_rect", None) is None
                else tuple(gui.control_panel_rect)
            ),
            "control_panel_collapsed": getattr(gui, "control_panel_collapsed", False),
            "sidebar_x": sidebar_x,
            "mouse_pos": mouse_pos,
        }
        if debug_input_active:
            logger.info("INPUT_DIAG: %s", diag)
        else:
            logger.debug("INPUT_DIAG: %s", diag)

        if debug_input_active:
            logger.info("INPUT_DIAG: KLTN_DEBUG_INPUT active - clearing overlays and ignore flags for this click")
            try:
                gui.preview_overlay_visible = False
                gui.show_solver_comparison_overlay = False
                gui.path_preview_dialog = None
                gui.control_panel_ignore_click_until = 0.0
                gui.control_panel_scroll_dragging = False
                gui._show_toast("Debug: overlays/ignore cleared", 1.2, "info")
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.exception("INPUT_DIAG: failed to clear debug state")
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("INPUT_DIAG: failure while computing diagnostics")

    try:
        sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
        if (
            getattr(gui, "preview_overlay_visible", False)
            and not getattr(gui, "preview_modal_enabled", False)
            and mouse_pos[0] < sidebar_x
        ):
            logger.info("Click on map detected while non-modal preview overlay active: dismissing overlay")
            try:
                gui.preview_overlay_visible = False
                gui.path_preview_dialog = None
                gui._show_toast("Preview dismissed (click)", 1.5, "info")
                gui._set_message("Preview dismissed", 1.5)
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                pass
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.exception("Error while checking/dismissing non-modal preview overlay")

    try:
        gui._last_mouse_event = {
            "type": "down",
            "pos": mouse_pos,
            "button": getattr(event, "button", None),
            "time": time_module.time(),
        }
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        gui._last_mouse_event = None

    if getattr(gui, "debug_click_log", None) is not None:
        gui.debug_click_log.insert(0, (mouse_pos, time_module.time()))
        if len(gui.debug_click_log) > 50:
            gui.debug_click_log.pop()

    if not focused:
        logger.info("Window does not have input focus; attempting to force focus")
        try:
            gui._force_focus()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("Force focus attempt failed")
        return mouse_pos, True

    return mouse_pos, False


def handle_mouse_button_down_event(gui, event, mouse_pos, pygame_module):
    """Handle panel interactions and map dragging on MOUSEBUTTONDOWN."""
    if gui.control_panel_enabled and gui.collapse_button_rect and gui.collapse_button_rect.collidepoint(mouse_pos):
        if not getattr(gui, "control_panel_animating", False):
            target_collapsed = not gui.control_panel_collapsed
            gui._start_toggle_panel_animation(target_collapsed)
        return True

    if gui.control_panel_enabled and gui.control_panel_rect and not gui.control_panel_collapsed:
        if (
            event.button == 1
            and getattr(gui, "control_panel_scroll_thumb_rect", None)
            and gui.control_panel_scroll_thumb_rect.collidepoint(mouse_pos)
        ):
            gui.control_panel_scroll_dragging = True
            gui.control_panel_scroll_drag_offset = mouse_pos[1] - gui.control_panel_scroll_thumb_rect.y
            return True

        if (
            event.button == 1
            and getattr(gui, "control_panel_scroll_track_rect", None)
            and gui.control_panel_scroll_track_rect.collidepoint(mouse_pos)
        ):
            tr = gui.control_panel_scroll_track_rect
            rel = mouse_pos[1] - tr.y
            ratio = max(0.0, min(1.0, rel / tr.height))
            gui.control_panel_scroll = int(ratio * getattr(gui, "control_panel_scroll_max", 0))
            return True

        title_bar_height = 45
        title_bar_rect = pygame_module.Rect(
            gui.control_panel_rect.x,
            gui.control_panel_rect.y,
            gui.control_panel_rect.width,
            title_bar_height,
        )
        if title_bar_rect.collidepoint(mouse_pos) and not gui.collapse_button_rect.collidepoint(mouse_pos):
            gui.dragging_panel = True
            gui.drag_panel_offset = (
                mouse_pos[0] - gui.control_panel_rect.x,
                mouse_pos[1] - gui.control_panel_rect.y,
            )
            return True

        edge_threshold = 8
        mx, my = mouse_pos
        rect = gui.control_panel_rect
        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
            gui.resizing_panel = True
            gui.resize_edge = "left"
            return True
        if abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
            gui.resizing_panel = True
            gui.resize_edge = "right"
            return True
        if abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
            gui.resizing_panel = True
            gui.resize_edge = "bottom"
            return True

    if gui.control_panel_enabled and gui._handle_control_panel_click(mouse_pos, event.button, "down"):
        return True

    if event.button == 1:
        if not gui._handle_minimap_click(mouse_pos):
            sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
            if mouse_pos[0] < sidebar_x and not (
                gui.control_panel_enabled
                and getattr(gui, "control_panel_rect", None)
                and gui.control_panel_rect.collidepoint(mouse_pos)
            ):
                gui.dragging = True
                gui.dragging_button = 1
                gui.drag_start = event.pos
    elif event.button == 2:
        gui.dragging = True
        gui.dragging_button = 2
        gui.drag_start = event.pos

    return False


def handle_mouse_button_up_event(gui, event, pygame_module, time_module, logger):
    """Handle MOUSEBUTTONUP bookkeeping and panel/click release logic."""
    mouse_pos = getattr(event, "pos", pygame_module.mouse.get_pos())
    try:
        focused = pygame_module.mouse.get_focused()
        grabbed = pygame_module.event.get_grab()
        pressed = pygame_module.mouse.get_pressed()
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        focused = False
        grabbed = False
        pressed = None

    logger.debug(
        "MOUSEBUTTONUP at %s (button=%s) focused=%s grabbed=%s pressed=%s",
        mouse_pos,
        getattr(event, "button", None),
        focused,
        grabbed,
        pressed,
    )

    try:
        gui._last_mouse_event = {
            "type": "up",
            "pos": mouse_pos,
            "button": getattr(event, "button", None),
            "time": time_module.time(),
        }
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        pass

    if gui.dragging_panel:
        gui.dragging_panel = False
    if gui.resizing_panel:
        gui.resizing_panel = False
        gui.resize_edge = None
    if getattr(gui, "control_panel_scroll_dragging", False):
        gui.control_panel_scroll_dragging = False

    if gui.control_panel_enabled and gui._handle_control_panel_click(mouse_pos, event.button, "up"):
        return True

    if event.button == 2 or (
        hasattr(gui, "dragging_button") and event.button == getattr(gui, "dragging_button")
    ):
        gui.dragging = False
        gui.dragging_button = None

    return False


def handle_mouse_motion_diagnostics(gui, event, pygame_module, time_module, logger):
    """Process throttled MOUSEMOTION diagnostics and track last motion event."""
    mouse_pos = event.pos
    try:
        focused = pygame_module.mouse.get_focused()
        grabbed = pygame_module.event.get_grab()
        buttons = pygame_module.mouse.get_pressed()
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        focused = False
        grabbed = False
        buttons = None

    now = time_module.time()
    last_log = getattr(gui, "_last_mouse_log_time", 0.0)
    suppressed = getattr(gui, "_mouse_motion_suppressed", 0)
    throttle = 0.05
    if (now - last_log) > throttle:
        logger.debug(
            "MOUSEMOTION at %s rel=%s buttons=%s focused=%s grabbed=%s suppressed=%d",
            mouse_pos,
            getattr(event, "rel", None),
            buttons,
            focused,
            grabbed,
            suppressed,
        )
        gui._last_mouse_log_time = now
        gui._mouse_motion_suppressed = 0
        gui._last_mouse_summary_time = getattr(gui, "_last_mouse_summary_time", 0.0)
    else:
        gui._mouse_motion_suppressed = suppressed + 1
        last_summary = getattr(gui, "_last_mouse_summary_time", 0.0)
        if (now - last_summary) > 1.0 and gui._mouse_motion_suppressed % 20 == 0:
            logger.debug(
                "MOUSEMOTION: still receiving motion events; suppressed=%d so far",
                gui._mouse_motion_suppressed,
            )
            gui._last_mouse_summary_time = now

    try:
        gui._last_mouse_event = {
            "type": "motion",
            "pos": mouse_pos,
            "rel": getattr(event, "rel", None),
            "time": time_module.time(),
        }
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        pass

    return mouse_pos


def handle_mouse_motion_event(gui, event, mouse_pos, pygame_module):
    """Handle panel drag/resize/scrollbar drag, map drag, and resize cursor updates."""
    if gui.dragging_panel:
        gui.control_panel_x = mouse_pos[0] - gui.drag_panel_offset[0]
        gui.control_panel_y = mouse_pos[1] - gui.drag_panel_offset[1]
        gui.control_panel_x = max(0, min(gui.control_panel_x, gui.screen_w - gui.control_panel_width))
        gui.control_panel_y = max(0, min(gui.control_panel_y, gui.screen_h - 100))
        gui._reposition_widgets(gui.control_panel_x, gui.control_panel_y)
        return

    if gui.resizing_panel and gui.control_panel_rect:
        if gui.resize_edge == "left":
            old_right = gui.control_panel_rect.right
            new_x = mouse_pos[0]
            new_width = old_right - new_x
            if gui.min_panel_width <= new_width <= gui.max_panel_width:
                gui.control_panel_width = new_width
                gui.control_panel_x = new_x
        elif gui.resize_edge == "right":
            new_width = mouse_pos[0] - gui.control_panel_rect.x
            if gui.min_panel_width <= new_width <= gui.max_panel_width:
                gui.control_panel_width = new_width
        elif gui.resize_edge == "bottom":
            new_height = mouse_pos[1] - gui.control_panel_rect.y
            if gui.min_panel_height <= new_height <= gui.screen_h - gui.control_panel_rect.y - 20:
                pass
        return

    if (
        getattr(gui, "control_panel_scroll_dragging", False)
        and getattr(gui, "control_panel_can_scroll", False)
        and getattr(gui, "control_panel_scroll_track_rect", None)
    ):
        track_rect = gui.control_panel_scroll_track_rect
        thumb_rect = getattr(gui, "control_panel_scroll_thumb_rect", None)
        if thumb_rect is None:
            return
        rel_y = mouse_pos[1] - track_rect.y
        max_move = track_rect.height - thumb_rect.height
        new_thumb_top = max(0, min(rel_y - getattr(gui, "control_panel_scroll_drag_offset", 0), max_move))
        if max_move > 0:
            ratio = new_thumb_top / max_move
            gui.control_panel_scroll = int(ratio * getattr(gui, "control_panel_scroll_max", 0))
            gui.control_panel_scroll = max(0, min(gui.control_panel_scroll, getattr(gui, "control_panel_scroll_max", 0)))
        return

    if gui.dragging:
        dx = gui.drag_start[0] - event.pos[0]
        dy = gui.drag_start[1] - event.pos[1]
        gui.view_offset_x += dx
        gui.view_offset_y += dy
        gui.drag_start = event.pos
        gui._clamp_view_offset()
        return

    if gui.control_panel_enabled and gui.control_panel_rect and not gui.control_panel_collapsed:
        edge_threshold = 8
        mx, my = mouse_pos
        rect = gui.control_panel_rect

        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
            pygame_module.mouse.set_cursor(pygame_module.SYSTEM_CURSOR_SIZEWE)
        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
            pygame_module.mouse.set_cursor(pygame_module.SYSTEM_CURSOR_SIZEWE)
        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
            pygame_module.mouse.set_cursor(pygame_module.SYSTEM_CURSOR_SIZENS)
        else:
            pygame_module.mouse.set_cursor(pygame_module.SYSTEM_CURSOR_ARROW)


def handle_mousewheel_event(gui, event, pygame_module, time_module):
    """Handle control-panel scroll momentum and map zoom on mouse wheel."""
    mouse_pos = pygame_module.mouse.get_pos()
    panel_rect = getattr(gui, "control_panel_rect", None)
    padding = getattr(gui, "debug_panel_click_padding", 0) if getattr(gui, "debug_control_panel", False) else 0
    panel_hit_rect = (
        pygame_module.Rect(
            panel_rect.x - padding,
            panel_rect.y,
            panel_rect.width + padding,
            panel_rect.height,
        )
        if panel_rect and padding
        else panel_rect
    )
    if (
        gui.control_panel_enabled
        and getattr(gui, "control_panel_can_scroll", False)
        and panel_hit_rect
        and panel_hit_rect.collidepoint(mouse_pos)
        and not gui.control_panel_collapsed
    ):
        wheel_power = getattr(gui, "control_panel_scroll_step", 20) * 12
        gui.control_panel_scroll_velocity += -event.y * wheel_power
        max_v = 2000
        gui.control_panel_scroll_velocity = max(-max_v, min(max_v, gui.control_panel_scroll_velocity))
        gui.control_panel_ignore_click_until = time_module.time() + 0.12
        return

    sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
    if mouse_pos[0] < sidebar_x:
        gui._change_zoom(event.y, center=mouse_pos)
    else:
        gui._change_zoom(event.y)
