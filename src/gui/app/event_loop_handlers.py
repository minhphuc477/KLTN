"""Helpers for gui_runner event loop orchestration."""

from __future__ import annotations


def poll_pygame_events(pygame_module, time_module, logger):
    """Fetch pending pygame events with slow-call diagnostics."""
    try:
        start_ts = time_module.time()
        events = pygame_module.event.get()
        duration = time_module.time() - start_ts
        if duration > 0.05:
            logger.debug("Slow event.get() detected: %.3fs", duration)
        return events
    except Exception:
        logger.exception("pygame.event.get() raised")
        return []


def run_input_focus_fallback(gui, pygame_module, time_module, logger, should_attempt_focus_fallback_fn):
    """Try to recover input focus in windowed mode when focus is lost."""
    try:
        focused = pygame_module.mouse.get_focused()
        now_ts = time_module.time()
        if should_attempt_focus_fallback_fn(
            gui.fullscreen,
            focused,
            now_ts,
            getattr(gui, "_last_ungrab_attempt", 0.0),
            cooldown_sec=2.0,
        ):
            logger.debug("Window lacks input focus; attempting to clear event grab and show cursor")
            try:
                pygame_module.event.set_grab(False)
            except Exception:
                logger.debug("Failed to clear event grab during fallback")
            try:
                pygame_module.mouse.set_visible(True)
            except Exception:
                logger.debug("Failed to set mouse visible during fallback")
            gui._last_ungrab_attempt = now_ts
    except Exception:
        logger.exception("Error during input focus fallback")


def clear_stale_preview_overlay(gui, logger):
    """Clear stale non-modal preview overlay state that can block interaction."""
    if not getattr(gui, "preview_overlay_visible", False):
        return
    if getattr(gui, "path_preview_dialog", None) or getattr(gui, "auto_path", None):
        return

    try:
        logger.warning("Clearing stale preview_overlay_visible (no dialog/path present) to restore input")
        gui.preview_overlay_visible = False
        gui.path_preview_dialog = None
        try:
            gui._set_message("Cleared stale preview overlay", 1.5)
        except Exception:
            pass
    except Exception:
        logger.exception("Failed to clear stale preview overlay")


def handle_window_focus_event(gui, event, pygame_module, logger):
    """Handle focus gain/loss events; returns True when consumed."""
    if event.type == getattr(pygame_module, "WINDOWFOCUSGAINED", None):
        logger.debug("WINDOWFOCUSGAINED: clearing event grab and showing mouse cursor")
        try:
            pygame_module.event.set_grab(False)
        except Exception:
            logger.debug("Could not clear event grab on focus gained")
        try:
            pygame_module.mouse.set_visible(True)
        except Exception:
            logger.debug("Could not set mouse visible on focus gained")
        try:
            gui._set_message("Window focused", 1.5)
        except Exception:
            pass
        return True

    if event.type == getattr(pygame_module, "WINDOWFOCUSLOST", None):
        logger.debug("WINDOWFOCUSLOST: pausing input interactions")
        try:
            gui._set_message("Window lost focus", 1.5)
        except Exception:
            pass
        return True

    return False


def handle_global_keydown_shortcuts(
    gui,
    event,
    pygame_module,
    time_module,
    logger,
    checkbox_widget_cls,
):
    """Handle global KEYDOWN diagnostics and shortcuts; returns True when consumed."""
    if event.type != pygame_module.KEYDOWN:
        return False

    try:
        gui._last_key_event = {
            "key": event.key,
            "mods": pygame_module.key.get_mods(),
            "time": time_module.time(),
        }
    except Exception:
        pass

    logger.debug("KEYDOWN key=%s mods=%s", event.key, pygame_module.key.get_mods())

    if event.key == pygame_module.K_o and (pygame_module.key.get_mods() & pygame_module.KMOD_CTRL):
        try:
            if getattr(gui, "preview_overlay_visible", False) or getattr(gui, "show_solver_comparison_overlay", False):
                gui.preview_overlay_visible = False
                gui.show_solver_comparison_overlay = False
                gui.path_preview_dialog = None
                gui._show_toast("Overlays hidden (Ctrl+O)", 2.0, "success")
                gui._set_message("Overlays hidden", 2.0)
            else:
                gui._show_toast("No overlays active", 1.5, "info")
        except Exception:
            logger.exception("Failed to toggle overlays")
        return True

    if event.key == pygame_module.K_F12:
        gui.debug_overlay_enabled = not getattr(gui, "debug_overlay_enabled", False)
        if gui.debug_overlay_enabled:
            gui._set_message("Debug overlay ON (F12 to toggle)")
        else:
            gui._set_message("Debug overlay OFF")
        return True

    if event.key == pygame_module.K_f:
        try:
            pygame_module.event.set_grab(False)
        except Exception:
            logger.debug("Failed to clear event grab via F key")
        try:
            pygame_module.mouse.set_visible(True)
        except Exception:
            logger.debug("Failed to set mouse visible via F key")
        try:
            gui._show_toast("Forced focus/ungrab (F)", 2.0, "info")
            gui._set_message("Forced focus/ungrab (F)")
        except Exception:
            pass
        return True

    if event.key == pygame_module.K_F12 and (pygame_module.key.get_mods() & pygame_module.KMOD_SHIFT):
        gui.debug_control_panel = not getattr(gui, "debug_control_panel", False)
        if gui.debug_control_panel:
            gui._set_message("Control panel debug ON (Shift+F12)")
        else:
            gui._set_message("Control panel debug OFF")
        return True

    if event.key in (pygame_module.K_PAGEUP, pygame_module.K_PAGEDOWN):
        if (
            gui.control_panel_enabled
            and getattr(gui, "control_panel_can_scroll", False)
            and getattr(gui, "control_panel_rect", None)
            and gui.control_panel_rect.collidepoint(pygame_module.mouse.get_pos())
            and not gui.control_panel_collapsed
        ):
            page_amount = max(1, gui.control_panel_rect.height - 32)
            if event.key == pygame_module.K_PAGEUP:
                gui.control_panel_scroll = max(0, int(gui.control_panel_scroll - page_amount))
            else:
                gui.control_panel_scroll = min(
                    getattr(gui, "control_panel_scroll_max", 0),
                    int(gui.control_panel_scroll + page_amount),
                )
            gui.control_panel_scroll_velocity = 0.0
            gui.control_panel_ignore_click_until = time_module.time() + 0.12
            return True

    if event.key == pygame_module.K_F11 and (pygame_module.key.get_mods() & pygame_module.KMOD_SHIFT):
        gui.debug_click_log = []
        gui._set_message("Debug log cleared")
        return True

    if event.key == pygame_module.K_t:
        gui.show_topology = not getattr(gui, "show_topology", False)
        for w in (gui.widget_manager.widgets if gui.widget_manager else []):
            if isinstance(w, checkbox_widget_cls) and getattr(w, "flag_name", "") == "show_topology":
                w.checked = gui.show_topology
        if gui.show_topology:
            cur = gui.maps[gui.current_map_idx]
            if not hasattr(cur, "graph") or not cur.graph:
                gui._set_message("Topology not available for this map", 3.0)
            else:
                gui._set_message("Topology overlay: ON", 2.0)
        else:
            gui._set_message("Topology overlay: OFF", 1.2)
        return True

    return False


def handle_preview_overlay_events(gui, event, pygame_module):
    """Handle preview overlay and path preview dialog interactions; returns True when consumed."""
    if getattr(gui, "preview_overlay_visible", False) and (
        gui.path_preview_dialog or getattr(gui, "auto_path", None)
    ):
        if event.type == pygame_module.KEYDOWN:
            if event.key == pygame_module.K_ESCAPE:
                gui.preview_overlay_visible = False
                gui.path_preview_dialog = None
                gui.message = "Path preview dismissed"
                return True
            if event.key == pygame_module.K_RETURN or event.key == pygame_module.K_SPACE:
                gui._execute_auto_solve_from_preview()
                return True

    if getattr(gui, "path_preview_mode", False) and getattr(gui, "path_preview_dialog", None):
        result = gui.path_preview_dialog.handle_input(event)
        if result == "start":
            gui._execute_auto_solve_from_preview()
            return True
        if result == "cancel":
            gui.path_preview_mode = False
            gui.preview_overlay_visible = True
            gui.message = "Path preview closed; overlay visible in sidebar/map (Enter to start or Esc to dismiss)"
            return True

    if (
        event.type == pygame_module.KEYDOWN
        and event.key == pygame_module.K_ESCAPE
        and getattr(gui, "show_solver_comparison_overlay", False)
    ):
        gui.show_solver_comparison_overlay = False
        gui._set_message("Solver comparison closed", 1.2)
        return True

    if (
        getattr(gui, "preview_overlay_visible", False)
        and event.type == pygame_module.MOUSEBUTTONDOWN
        and event.button == 1
    ):
        mouse_pos = event.pos
        if getattr(gui, "sidebar_start_button_rect", None) and gui.sidebar_start_button_rect.collidepoint(mouse_pos):
            gui._execute_auto_solve_from_preview()
            return True
        if getattr(gui, "sidebar_dismiss_button_rect", None) and gui.sidebar_dismiss_button_rect.collidepoint(mouse_pos):
            gui.preview_overlay_visible = False
            gui.path_preview_dialog = None
            gui.message = "Path preview dismissed"
            return True

    return False


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
    except Exception:
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
            except Exception:
                logger.exception("INPUT_DIAG: failed to clear debug state")
    except Exception:
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
            except Exception:
                pass
    except Exception:
        logger.exception("Error while checking/dismissing non-modal preview overlay")

    try:
        gui._last_mouse_event = {
            "type": "down",
            "pos": mouse_pos,
            "button": getattr(event, "button", None),
            "time": time_module.time(),
        }
    except Exception:
        gui._last_mouse_event = None

    if getattr(gui, "debug_click_log", None) is not None:
        gui.debug_click_log.insert(0, (mouse_pos, time_module.time()))
        if len(gui.debug_click_log) > 50:
            gui.debug_click_log.pop()

    if not focused:
        logger.info("Window does not have input focus; attempting to force focus")
        try:
            gui._force_focus()
        except Exception:
            logger.exception("Force focus attempt failed")
        return mouse_pos, True

    return mouse_pos, False


def handle_mouse_button_up_event(gui, event, pygame_module, time_module, logger):
    """Handle MOUSEBUTTONUP bookkeeping and panel/click release logic."""
    mouse_pos = getattr(event, "pos", pygame_module.mouse.get_pos())
    try:
        focused = pygame_module.mouse.get_focused()
        grabbed = pygame_module.event.get_grab()
        pressed = pygame_module.mouse.get_pressed()
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        pass

    return mouse_pos


def handle_keyup_event(gui, event, logger):
    """Handle KEYUP bookkeeping for continuous movement keys."""
    try:
        if event.key in getattr(gui, "keys_held", {}):
            gui.keys_held[event.key] = False
    except Exception:
        logger.debug("Failed to handle KEYUP for %r", getattr(event, "key", None))


def handle_keydown_event(
    gui,
    event,
    pygame_module,
    os_module,
    logger,
    checkbox_widget_cls,
    action_enum,
    running,
):
    """Handle KEYDOWN gameplay/UI controls and return updated running state."""
    if event.key == pygame_module.K_ESCAPE:
        if gui.fullscreen:
            gui._toggle_fullscreen()
        else:
            running = False

    elif event.key == pygame_module.K_F11:
        gui._toggle_fullscreen()

    elif event.key == pygame_module.K_h:
        if not gui.show_help:
            gui.show_heatmap = not gui.show_heatmap
            gui.feature_flags["show_heatmap"] = gui.show_heatmap
            if gui.renderer:
                gui.renderer.show_heatmap = gui.show_heatmap
            if gui.widget_manager:
                for widget in gui.widget_manager.widgets:
                    if (
                        isinstance(widget, checkbox_widget_cls)
                        and hasattr(widget, "flag_name")
                        and widget.flag_name == "show_heatmap"
                    ):
                        widget.checked = gui.show_heatmap
            gui.message = f"Heatmap: {'ON' if gui.show_heatmap else 'OFF'}"

    elif event.key == pygame_module.K_F1:
        gui.show_help = not gui.show_help

    elif event.key == pygame_module.K_TAB:
        if gui.control_panel_enabled and not getattr(gui, "control_panel_animating", False):
            target_collapsed = not gui.control_panel_collapsed
            gui._start_toggle_panel_animation(target_collapsed)

    elif event.key == pygame_module.K_F7:
        try:
            pos = pygame_module.mouse.get_pos()
            logger.info(
                "DIAG DUMP (F7): mouse_pos=%s control_panel_rect=%s scroll=%s",
                pos,
                getattr(gui, "control_panel_rect", None),
                getattr(gui, "control_panel_scroll", 0),
            )
            try:
                gui._dump_control_panel_widget_state(pos)
            except Exception:
                logger.exception("F7: _dump_control_panel_widget_state failed")
        except Exception:
            logger.exception("F7 diagnostic failed")

    elif event.key == pygame_module.K_F8:
        try:
            gui.debug_control_panel = not getattr(gui, "debug_control_panel", False)
            gui.debug_panel_click_padding = (
                int(os_module.environ.get("KLTN_DEBUG_PANEL_PADDING", "40"))
                if gui.debug_control_panel
                else 0
            )
            gui._show_toast(
                f"Debug control panel {'ON' if gui.debug_control_panel else 'OFF'}",
                1.6,
                "info",
            )
            logger.info(
                "Toggled debug_control_panel=%s padding=%s",
                gui.debug_control_panel,
                gui.debug_panel_click_padding,
            )
        except Exception:
            logger.exception("Failed to toggle debug control panel")

    elif event.key == pygame_module.K_m:
        gui.show_minimap = not gui.show_minimap
        gui.feature_flags["show_minimap"] = gui.show_minimap
        if gui.widget_manager:
            for widget in gui.widget_manager.widgets:
                if (
                    isinstance(widget, checkbox_widget_cls)
                    and hasattr(widget, "flag_name")
                    and widget.flag_name == "show_minimap"
                ):
                    widget.checked = gui.show_minimap
        gui.message = f"Minimap: {'ON' if gui.show_minimap else 'OFF'}"

    elif event.key == pygame_module.K_RIGHTBRACKET or event.key == pygame_module.K_PERIOD:
        gui.speed_index = min(len(gui.speed_levels) - 1, gui.speed_index + 1)
        gui.speed_multiplier = gui.speed_levels[gui.speed_index]
        gui.message = f"Speed: {gui.speed_multiplier}x"

    elif event.key == pygame_module.K_LEFTBRACKET or event.key == pygame_module.K_COMMA:
        gui.speed_index = max(0, gui.speed_index - 1)
        gui.speed_multiplier = gui.speed_levels[gui.speed_index]
        gui.message = f"Speed: {gui.speed_multiplier}x"

    elif event.key == pygame_module.K_SPACE:
        gui._start_auto_solve()

    elif event.key == pygame_module.K_r:
        gui._load_current_map()
        gui._center_view()
        if gui.effects:
            gui.effects.clear()
        gui.step_count = 0
        gui.message = "Map Reset"

    elif event.key == pygame_module.K_n:
        gui._next_map()

    elif event.key == pygame_module.K_p:
        gui._prev_map()

    elif event.key == pygame_module.K_PLUS or event.key == pygame_module.K_EQUALS:
        gui._change_zoom(1)

    elif event.key == pygame_module.K_MINUS:
        gui._change_zoom(-1)

    elif event.key == pygame_module.K_0:
        gui.zoom_idx = gui.DEFAULT_ZOOM_IDX
        gui.TILE_SIZE = gui.ZOOM_LEVELS[gui.zoom_idx]
        gui._load_assets()
        gui._center_view()
        gui.message = "Zoom reset to default"

    elif event.key == pygame_module.K_f:
        gui._auto_fit_zoom()
        gui.message = f"Auto-fit: {gui.TILE_SIZE}px"

    elif event.key == pygame_module.K_c:
        gui._center_on_player()

    elif event.key == pygame_module.K_l:
        ok = gui.load_visual_map(os_module.path.join(os_module.getcwd(), "screenshot.png"))
        if not ok:
            gui.message = "Failed to load ./screenshot.png"

    elif event.key in gui.keys_held and not gui.auto_mode:
        gui.keys_held[event.key] = True
        gui.move_timer = 0.0

    elif not gui.auto_mode:
        keys = pygame_module.key.get_pressed()
        action = None
        if keys[pygame_module.K_UP] and keys[pygame_module.K_LEFT]:
            action = action_enum.UP_LEFT
        elif keys[pygame_module.K_UP] and keys[pygame_module.K_RIGHT]:
            action = action_enum.UP_RIGHT
        elif keys[pygame_module.K_DOWN] and keys[pygame_module.K_LEFT]:
            action = action_enum.DOWN_LEFT
        elif keys[pygame_module.K_DOWN] and keys[pygame_module.K_RIGHT]:
            action = action_enum.DOWN_RIGHT
        elif keys[pygame_module.K_UP]:
            action = action_enum.UP
        elif keys[pygame_module.K_DOWN]:
            action = action_enum.DOWN
        elif keys[pygame_module.K_LEFT]:
            action = action_enum.LEFT
        elif keys[pygame_module.K_RIGHT]:
            action = action_enum.RIGHT

        if action is not None:
            gui._manual_step(action)
            gui._center_on_player()

    return running


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
            except Exception:
                logger.exception("VIDEORESIZE: display reinit failed")
        else:
            gui.screen = screen
            try:
                gui.screen_w, gui.screen_h = gui.screen.get_size()
            except Exception:
                pass

        try:
            gui._load_assets()
            gui._render()
            try:
                pygame_module.display.flip()
            except Exception:
                logger.exception("Flip failed after VIDEORESIZE")
        except Exception:
            logger.exception("Failed to refresh UI after VIDEORESIZE")

    if gui.control_panel_enabled:
        gui._update_control_panel_positions()


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
