"""Keyboard event handlers for GUI event loop."""

from __future__ import annotations


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