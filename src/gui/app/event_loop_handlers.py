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
