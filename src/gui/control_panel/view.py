"""Control-panel layout/render helpers extracted from gui_runner."""

from __future__ import annotations

from typing import Any


def update_control_panel_positions(
    gui: Any,
    pygame: Any,
    logger: Any,
    *,
    widgets_available: bool,
    checkbox_widget_cls: Any,
    dropdown_widget_cls: Any,
    button_widget_cls: Any,
    zoom_labels: list,
    difficulty_names: list,
    algorithm_names: list,
) -> None:
    """Update control panel/widget positions, rebuilding widgets only when needed."""
    if not widgets_available or not gui.widget_manager:
        return

    collapsed_width = 40
    max_allowed_panel_width = max(
        collapsed_width,
        min(
            gui.control_panel_width_current,
            gui.max_panel_width,
            max(120, min(gui.SIDEBAR_WIDTH * 2, max(120, gui.screen_w - gui.SIDEBAR_WIDTH - 40))),
        ),
    )
    panel_width = int(max_allowed_panel_width)

    try:
        original_width = int(max(collapsed_width, min(gui.control_panel_width_current, gui.max_panel_width)))
        if panel_width != original_width:
            logger.debug(
                "Control panel width clamped from %d to %d due to SIDEBAR_WIDTH=%d",
                original_width,
                panel_width,
                gui.SIDEBAR_WIDTH,
            )
    except Exception:
        pass

    sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
    if gui.control_panel_x is not None and gui.control_panel_y is not None:
        panel_x = gui.control_panel_x
        panel_y = gui.control_panel_y
    else:
        panel_x = sidebar_x - panel_width - 10
        panel_y = gui.minimap_size + 20 if gui.show_minimap else 10

    min_x = 10
    max_x = max(min_x, sidebar_x - panel_width - 10)
    panel_x = max(min_x, min(panel_x, max_x))

    max_available_height = gui.screen_h - panel_y - gui.HUD_HEIGHT - 20
    min_panel_height = 120
    if max_available_height < min_panel_height:
        panel_height = min_panel_height
    else:
        panel_height = min(max_available_height, 700)

    gui.control_panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

    button_size = 32
    gui.collapse_button_rect = pygame.Rect(
        panel_x + panel_width - 34,
        panel_y + 4,
        button_size,
        button_size,
    )

    if panel_width <= collapsed_width + 8:
        return

    widgets_exist = hasattr(gui, "widget_manager") and gui.widget_manager and len(gui.widget_manager.widgets) > 0
    if widgets_exist:
        gui._reposition_widgets(panel_x, panel_y)
        return

    saved_dropdown_state = None
    if hasattr(gui, "widget_manager") and gui.widget_manager:
        saved_dropdown_state = gui.widget_manager.snapshot_dropdown_state()
        gui.widget_manager.widgets.clear()

    margin_left = 12
    margin_top = 48
    checkbox_spacing = 26
    dropdown_spacing = 44
    section_gap = 18

    y_offset = panel_y + margin_top
    x_offset = panel_x + margin_left

    checkbox_labels = [
        ("solver_comparison", "Solver Comparison"),
        ("parallel_search", "Parallel Search"),
        ("multi_goal", "Multi-Goal Pathfinding"),
        ("ml_heuristic", "ML Heuristic"),
        ("dstar_lite", "D* Lite Replanning"),
        ("show_heatmap", "Show Heatmap Overlay"),
        ("show_path", "Show Path Overlay"),
        ("show_map_elites", "Show MAP-Elites Overlay"),
        ("show_topology", "Show Topology Overlay"),
        ("show_topology_legend", "Topology Legend (details)"),
        ("show_minimap", "Show Minimap"),
        ("diagonal_movement", "Diagonal Movement"),
        ("use_jps", "Use Jump Point Search (JPS)"),
        ("show_jps_overlay", "Show JPS Overlay"),
        ("speedrun_mode", "Speedrun Mode"),
        ("strict_original_mode", "Strict Original LoZ Rules"),
        ("dynamic_difficulty", "Dynamic Difficulty"),
        ("force_grid", "Force Grid Solver"),
        ("enable_prechecks", "Enable Prechecks (fast checks before solve)"),
        ("auto_prune_on_precheck", "Auto-Prune Dead-Ends on Precheck"),
        ("priority_tie_break", "Priority: Tie-Break by Locks"),
        ("priority_key_boost", "Priority: Key-Pickup Boost"),
        ("enable_ara", "Enable ARA* (weighted A*)"),
        ("persist_dropdown_on_select", "Keep dropdown open after select"),
    ]

    for flag_name, label in checkbox_labels:
        checkbox = checkbox_widget_cls(
            (x_offset, y_offset),
            label,
            checked=gui.feature_flags.get(flag_name, False),
        )
        setattr(checkbox, "flag_name", flag_name)
        gui.widget_manager.add_widget(checkbox)
        y_offset += checkbox_spacing

    y_offset += section_gap

    floor_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Floor",
        ["Floor 1", "Floor 2", "Floor 3"],
        selected=gui.current_floor - 1,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(floor_dropdown, "control_name", "floor")
    gui.widget_manager.add_widget(floor_dropdown)
    y_offset += dropdown_spacing

    zoom_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Zoom",
        zoom_labels,
        selected=gui.zoom_level_idx,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(zoom_dropdown, "control_name", "zoom")
    gui.widget_manager.add_widget(zoom_dropdown)
    y_offset += dropdown_spacing

    ara_options = ["1.0", "1.25", "1.5", "2.0"]
    try:
        ara_value = float(getattr(gui, "ara_weight", 1.0))
        ara_selected = min(
            range(len(ara_options)),
            key=lambda idx: abs(float(ara_options[idx]) - ara_value),
        )
    except Exception:
        ara_selected = 0
    ara_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "ARA* weight",
        ara_options,
        selected=ara_selected,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(ara_dropdown, "control_name", "ara_weight")
    gui.widget_manager.add_widget(ara_dropdown)
    y_offset += dropdown_spacing

    difficulty_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Difficulty",
        difficulty_names,
        selected=gui.difficulty_idx,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(difficulty_dropdown, "control_name", "difficulty")
    gui.widget_manager.add_widget(difficulty_dropdown)
    y_offset += dropdown_spacing

    presets_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Presets",
        gui.presets,
        selected=gui.current_preset_idx,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(presets_dropdown, "control_name", "presets")
    gui.widget_manager.add_widget(presets_dropdown)
    y_offset += dropdown_spacing

    algorithm_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Solver",
        algorithm_names,
        selected=gui.algorithm_idx,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(algorithm_dropdown, "control_name", "algorithm")
    gui.widget_manager.add_widget(algorithm_dropdown)
    y_offset += dropdown_spacing

    rep_options = ["Hybrid (Graph+Tile)", "Tile Only", "Graph Only"]
    rep_to_idx = {"hybrid": 0, "tile": 1, "graph": 2}
    rep_selected = rep_to_idx.get(str(getattr(gui, "search_representation", "hybrid")).lower(), 0)
    representation_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Search Space",
        rep_options,
        selected=rep_selected,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(representation_dropdown, "control_name", "representation")
    gui.widget_manager.add_widget(representation_dropdown)
    y_offset += dropdown_spacing

    threshold_dropdown = dropdown_widget_cls(
        (x_offset, y_offset),
        "Apply Threshold",
        ["0.70", "0.75", "0.80", "0.85", "0.90"],
        selected=3,
        keep_open_on_select=gui.feature_flags.get("persist_dropdown_on_select", False),
    )
    setattr(threshold_dropdown, "control_name", "match_threshold")
    gui.widget_manager.add_widget(threshold_dropdown)
    gui.match_apply_threshold = float(threshold_dropdown.options[threshold_dropdown.selected])
    y_offset += dropdown_spacing

    y_offset += section_gap

    try:
        if saved_dropdown_state is not None:
            gui.widget_manager.apply_dropdown_state(saved_dropdown_state)
    except Exception:
        pass

    button_width = 125
    button_height = 30
    buttons_per_row = 2
    button_h_spacing = 8
    button_v_spacing = 8

    primary_buttons = [
        ("Start Auto-Solve", gui._start_auto_solve),
        ("Stop", gui._stop_auto_solve),
        ("Generate Dungeon", gui._generate_dungeon),
        ("AI Generate", gui._generate_ai_dungeon),
        ("Reset", gui._reset_map),
    ]

    secondary_buttons = [
        ("Path Preview", gui._show_path_preview),
        ("Clear Path", gui._clear_path),
        ("Export Route", gui._export_route),
        ("Load Route", gui._load_route),
        ("Open Temp Folder", gui._open_temp_folder),
        ("Delete Temp Files", gui._delete_temp_files),
        ("Export Topology", gui._export_topology),
        ("Compare Solvers", gui._run_solver_comparison),
        ("Match Missing Nodes", gui._match_missing_nodes),
        ("Apply Tentative Matches", gui._apply_tentative_matches),
        ("Undo Last Match", gui._undo_last_match),
        ("Undo Prune", gui._undo_prune),
        ("Run MAP-Elites", gui._start_map_elites),
    ]

    button_start_y = y_offset
    for i, (label, callback) in enumerate(primary_buttons):
        row = i // buttons_per_row
        col = i % buttons_per_row
        button_x = x_offset + col * (button_width + button_h_spacing)
        button_y = button_start_y + row * (button_height + button_v_spacing)

        button = button_widget_cls(
            (button_x, button_y),
            label,
            callback,
            width=button_width,
            height=button_height,
        )
        gui.widget_manager.add_widget(button)

    y_offset = button_start_y + (len(primary_buttons) // buttons_per_row) * (button_height + button_v_spacing) + 12

    for i, (label, callback) in enumerate(secondary_buttons):
        row = i // buttons_per_row
        col = i % buttons_per_row
        button_x = x_offset + col * (button_width + button_h_spacing)
        button_y = y_offset + row * (button_height + button_v_spacing)

        button = button_widget_cls(
            (button_x, button_y),
            label,
            callback,
            width=button_width,
            height=button_height,
        )
        gui.widget_manager.add_widget(button)

    try:
        gui._reposition_widgets(panel_x, panel_y)
    except Exception:
        pass

    max_widget_bottom = 0
    min_widget_top = 10**9
    for w in gui.widget_manager.widgets:
        if hasattr(w, "full_rect") and getattr(w, "full_rect") is not None:
            top = int(getattr(w, "full_rect").top)
        elif hasattr(w, "rect") and getattr(w, "rect") is not None:
            top = int(getattr(w, "rect").top)
        else:
            top = panel_y

        bottoms = []
        if hasattr(w, "dropdown_rect") and getattr(w, "dropdown_rect") is not None:
            try:
                bottoms.append(int(getattr(w, "dropdown_rect").bottom))
            except Exception:
                pass
        if hasattr(w, "full_rect") and getattr(w, "full_rect") is not None:
            bottoms.append(int(getattr(w, "full_rect").bottom))
        if hasattr(w, "rect") and getattr(w, "rect") is not None:
            bottoms.append(int(getattr(w, "rect").bottom))

        bottom = max(bottoms) if bottoms else top

        min_widget_top = min(min_widget_top, top)
        max_widget_bottom = max(max_widget_bottom, bottom)

    extra_top = max(0, panel_y - min_widget_top)
    content_height = max_widget_bottom - panel_y + 12 + extra_top if max_widget_bottom > 0 else min_panel_height
    gui.control_panel_content_height = content_height

    if content_height > max_available_height:
        panel_height = min(max_available_height, 700)
        gui.control_panel_can_scroll = True
        gui.control_panel_scroll = getattr(gui, "control_panel_scroll", 0)
        gui.control_panel_scroll = max(0, min(gui.control_panel_scroll, content_height - panel_height))
        gui.control_panel_scroll_max = max(0, content_height - panel_height)
    else:
        panel_height = min(max_available_height, max(content_height, min_panel_height))
        gui.control_panel_can_scroll = False
        gui.control_panel_scroll = 0
        gui.control_panel_scroll_max = 0

    gui.control_panel_rect.height = panel_height
    gui.collapse_button_rect.y = panel_y + 4
    gui._reposition_widgets(panel_x, panel_y)


def reposition_widgets(
    gui: Any,
    panel_x: int,
    panel_y: int,
    *,
    checkbox_widget_cls: Any,
    dropdown_widget_cls: Any,
    button_widget_cls: Any,
) -> None:
    """Reposition existing widgets when panel is dragged without rebuilding."""
    if not gui.widget_manager or not gui.widget_manager.widgets:
        return

    margin_left = 12
    button_size = 28
    button_margin = 6
    margin_top = button_margin + max(button_size, gui.font.get_height()) + 8
    item_spacing = 44
    section_gap = 20

    current_y = panel_y + margin_top

    checkbox_idx = 0
    dropdown_idx = 0
    button_idx = 0

    button_width = 125
    button_height = 30
    buttons_per_row = 2
    button_h_spacing = 8
    button_v_spacing = 8

    for widget in gui.widget_manager.widgets:
        if isinstance(widget, checkbox_widget_cls):
            setattr(widget, "pos", (panel_x + margin_left, current_y + checkbox_idx * item_spacing))
            checkbox_idx += 1
        elif isinstance(widget, dropdown_widget_cls):
            if checkbox_idx > 0 and dropdown_idx == 0:
                current_y += checkbox_idx * item_spacing + section_gap
            setattr(widget, "pos", (panel_x + margin_left, current_y + dropdown_idx * item_spacing))
            dropdown_idx += 1
        elif isinstance(widget, button_widget_cls):
            if dropdown_idx > 0 and button_idx == 0:
                current_y += dropdown_idx * item_spacing + section_gap

            row = button_idx // buttons_per_row
            col = button_idx % buttons_per_row
            setattr(
                widget,
                "pos",
                (
                    panel_x + margin_left + col * (button_width + button_h_spacing),
                    current_y + row * (button_height + button_v_spacing),
                ),
            )
            button_idx += 1


def dump_control_panel_widget_state(
    gui: Any,
    mouse_pos: tuple,
    *,
    logger: Any,
    debug_input_active: bool,
) -> None:
    """Emit widget hit-test diagnostics for control-panel click debugging."""
    try:
        scroll_offset = getattr(gui, "control_panel_scroll", 0) if getattr(gui, "control_panel_can_scroll", False) else 0
        panel_rect = getattr(gui, "control_panel_rect", None)
        panel_top = panel_rect.y if panel_rect is not None else None
        header_height = gui.font.get_height() + 12
        sc_pos = (mouse_pos[0], mouse_pos[1] + scroll_offset)
        if debug_input_active:
            logger.info(
                "DUMP_WIDGETS: mouse_pos=%s sc_pos=%s scroll=%s panel_top=%s header_h=%s",
                mouse_pos,
                sc_pos,
                scroll_offset,
                panel_top,
                header_height,
            )
        else:
            logger.debug(
                "DUMP_WIDGETS: mouse_pos=%s sc_pos=%s scroll=%s panel_top=%s header_h=%s",
                mouse_pos,
                sc_pos,
                scroll_offset,
                panel_top,
                header_height,
            )

        for w in gui.widget_manager.widgets:
            fr = getattr(w, "full_rect", getattr(w, "rect", None))
            rr = getattr(w, "rect", None)
            dropdown_r = getattr(w, "dropdown_rect", None)
            try:
                fr_tuple = (fr.x, fr.y, fr.width, fr.height) if fr is not None else None
            except Exception:
                fr_tuple = None
            try:
                r_tuple = (rr.x, rr.y, rr.width, rr.height) if rr is not None else None
            except Exception:
                r_tuple = None
            try:
                dr_tuple = (dropdown_r.x, dropdown_r.y, dropdown_r.width, dropdown_r.height) if dropdown_r is not None else None
            except Exception:
                dr_tuple = None

            contains_mouse = bool(fr and fr.collidepoint(mouse_pos))
            contains_sc = bool(fr and fr.collidepoint(sc_pos))
            rect_contains_mouse = bool(rr and rr.collidepoint(mouse_pos))
            rect_contains_sc = bool(rr and rr.collidepoint(sc_pos))
            label = getattr(w, "label", getattr(w, "control_name", None) or w.__class__.__name__)

            if debug_input_active:
                logger.info(
                    "WIDGET: name=%r full_rect=%s rect=%s dropdown=%s contains_mouse=%s contains_sc=%s rect_contains_mouse=%s rect_contains_sc=%s type=%s",
                    label,
                    fr_tuple,
                    r_tuple,
                    dr_tuple,
                    contains_mouse,
                    contains_sc,
                    rect_contains_mouse,
                    rect_contains_sc,
                    w.__class__.__name__,
                )
            else:
                logger.debug(
                    "WIDGET: name=%r full_rect=%s rect=%s dropdown=%s contains_mouse=%s contains_sc=%s rect_contains_mouse=%s rect_contains_sc=%s type=%s",
                    label,
                    fr_tuple,
                    r_tuple,
                    dr_tuple,
                    contains_mouse,
                    contains_sc,
                    rect_contains_mouse,
                    rect_contains_sc,
                    w.__class__.__name__,
                )
    except Exception:
        logger.exception("Failed to dump control panel widget state")


def render_control_panel(
    gui: Any,
    surface: Any,
    *,
    pygame: Any,
    logger: Any,
    dropdown_widget_cls: Any,
) -> None:
    """Render control panel widgets, scroll chrome, dropdown menus, and tooltip layer."""
    if not gui.control_panel_enabled or not gui.widget_manager:
        return

    logger.debug(
        "_render_control_panel: width_current=%s, collapsed=%s, animating=%s",
        gui.control_panel_width_current,
        gui.control_panel_collapsed,
        getattr(gui, "control_panel_animating", False),
    )

    collapsed_width = 40

    try:
        gui._update_control_panel_positions()
    except Exception:
        pass

    if getattr(gui, "control_panel_rect", None):
        panel_rect = gui.control_panel_rect
        panel_x, panel_y, panel_width, panel_height = panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height
    else:
        panel_width = int(max(collapsed_width, min(gui.control_panel_width_current, gui.max_panel_width)))
        panel_width = max(collapsed_width, min(panel_width, gui.max_panel_width))

        sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
        if gui.control_panel_x is not None and gui.control_panel_y is not None:
            panel_x = gui.control_panel_x
            panel_y = gui.control_panel_y
        else:
            panel_x = sidebar_x - panel_width - 10
            panel_y = 10

        min_x = 10
        max_x = max(min_x, sidebar_x - panel_width - 10)
        panel_x = max(min_x, min(panel_x, max_x))
        panel_y = max(10, min(panel_y, gui.screen_h - 150))

        max_available_height = gui.screen_h - panel_y - gui.HUD_HEIGHT - 20
        min_panel_height = 120
        if max_available_height < min_panel_height:
            panel_height = min_panel_height
        else:
            panel_height = min(max_available_height, 700)

        if panel_width <= 0 or panel_height <= 0:
            return
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        gui.control_panel_rect = panel_rect

    panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    bg_rect = pygame.Rect(0, 0, panel_width, panel_height)
    pygame.draw.rect(panel_surf, (40, 45, 60, 255), bg_rect, border_radius=8)
    pygame.draw.rect(panel_surf, (60, 60, 80, 255), bg_rect, 2, border_radius=8)

    alpha = 255
    if getattr(gui, "control_panel_animating", False):
        a_from = gui.control_panel_anim_from
        a_to = gui.control_panel_anim_to
        denom = (a_to - a_from) if abs(a_to - a_from) > 1e-6 else 1.0
        progress = max(0.0, min(1.0, (gui.control_panel_width_current - a_from) / denom))
        ease = progress * progress * (3 - 2 * progress)
        alpha = int(255 * ease)

    if not getattr(gui, "control_panel_rect", None):
        gui.control_panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    if not getattr(gui, "collapse_button_rect", None):
        gui.collapse_button_rect = pygame.Rect(panel_x + panel_width - 28 - 6, panel_y + 6, 28, 28)

    mouse_pos = pygame.mouse.get_pos()
    is_hovering = gui.collapse_button_rect.collidepoint(mouse_pos)

    if panel_width <= collapsed_width + 8:
        if is_hovering:
            button_color = (80, 120, 180)
            border_color = (150, 200, 255)
        else:
            button_color = (60, 80, 120)
            border_color = (100, 150, 200)
        pygame.draw.rect(surface, button_color, gui.collapse_button_rect, border_radius=4)
        pygame.draw.rect(surface, border_color, gui.collapse_button_rect, 2, border_radius=4)
        if getattr(gui, "control_panel_animating", False):
            button_text = ">" if gui.control_panel_target_collapsed else "<"
        else:
            button_text = "<" if not gui.control_panel_collapsed else ">"
        button_surf = gui.font.render(button_text, True, (200, 220, 255))
        button_rect = button_surf.get_rect(center=gui.collapse_button_rect.center)
        surface.blit(button_surf, button_rect)
        return

    gui._reposition_widgets(panel_x, panel_y)
    backups = [(w, getattr(w, "pos", None)) for w in gui.widget_manager.widgets]

    content_alpha = 255
    if getattr(gui, "control_panel_animating", False):
        a_from = gui.control_panel_anim_from
        a_to = gui.control_panel_anim_to
        denom = (a_to - a_from) if abs(a_to - a_from) > 1e-6 else 1.0
        progress = max(0.0, min(1.0, (gui.control_panel_width_current - a_from) / denom))
        ease = progress * progress * (3 - 2 * progress)
        if ease >= 0.98:
            content_alpha = 255
        elif ease <= 0.02:
            content_alpha = 0
        else:
            content_alpha = int(255 * ease)

    for widget, orig_pos in backups:
        try:
            if orig_pos is None:
                continue
            full_rect = getattr(widget, "full_rect", widget.rect)
            dropdown_rect = getattr(widget, "dropdown_rect", None)
            scroll_offset = getattr(gui, "control_panel_scroll", 0) if getattr(gui, "control_panel_can_scroll", False) else 0

            local_x = full_rect.x - panel_x
            local_y = full_rect.y - panel_y - scroll_offset

            target_w = min(
                panel_width - 24,
                max(full_rect.width, dropdown_rect.width if dropdown_rect is not None else 0, widget.rect.width),
            )
            target_h = full_rect.height

            header_height = gui.font.get_height() + 12
            if local_y + target_h < header_height or local_y > panel_height:
                continue

            temp_surf = pygame.Surface((max(1, target_w), max(1, target_h)), pygame.SRCALPHA)

            label_offset = max(0, full_rect.height - widget.rect.height)
            widget.pos = (0, int(label_offset))
            widget.render(temp_surf)

            if content_alpha < 255:
                temp_surf.set_alpha(content_alpha)

            panel_surf.blit(temp_surf, (local_x, local_y))

        except Exception as e:
            logger.warning("Per-widget render failed: %s", e)

    if getattr(gui, "control_panel_can_scroll", False):
        track_w = 10
        track_margin = 8
        track_local_x = panel_width - track_w - track_margin
        track_local_y = 16
        track_h = panel_height - 32
        track_rect = pygame.Rect(track_local_x, track_local_y, track_w, track_h)
        pygame.draw.rect(panel_surf, (60, 65, 80, 200), track_rect, border_radius=6)

        header_height = gui.font.get_height() + 12
        visible_h = max(10, panel_height - header_height - 16)
        content_h = getattr(gui, "control_panel_content_height", visible_h)
        max_scroll = max(content_h - visible_h, 0)
        thumb_h = max(int((visible_h / content_h) * track_h) if content_h > 0 else track_h, 20)
        if max_scroll > 0:
            thumb_y_local = track_local_y + int((getattr(gui, "control_panel_scroll", 0) / max_scroll) * (track_h - thumb_h))
        else:
            thumb_y_local = track_local_y
        thumb_rect = pygame.Rect(track_local_x + 1, thumb_y_local, track_w - 2, thumb_h)
        pygame.draw.rect(panel_surf, (100, 130, 180, 220), thumb_rect, border_radius=6)

        gui.control_panel_scroll_track_rect = pygame.Rect(panel_x + track_rect.x, panel_y + track_rect.y, track_rect.width, track_rect.height)
        gui.control_panel_scroll_thumb_rect = pygame.Rect(panel_x + thumb_rect.x, panel_y + thumb_rect.y, thumb_rect.width, thumb_rect.height)
        gui.control_panel_scroll_max = max_scroll
    else:
        gui.control_panel_scroll_track_rect = None
        gui.control_panel_scroll_thumb_rect = None
        gui.control_panel_scroll_max = 0

    header_height = gui.font.get_height() + 12
    header_surf = pygame.Surface((panel_width, header_height), pygame.SRCALPHA)
    pygame.draw.rect(header_surf, (35, 40, 55, 230), pygame.Rect(0, 0, panel_width, header_height), border_radius=0)
    features_title = gui.font.render("FEATURES", True, (100, 200, 100))
    header_surf.blit(features_title, (12, (header_height - gui.font.get_height()) // 2))
    panel_surf.blit(header_surf, (0, 0))

    if alpha < 255:
        panel_surf.set_alpha(alpha)
    surface.blit(panel_surf, (panel_x, panel_y))
    if alpha < 255:
        panel_surf.set_alpha(255)

    for widget, orig_pos in backups:
        try:
            if orig_pos is not None:
                widget.pos = orig_pos
        except Exception:
            pass

    if is_hovering:
        button_color = (80, 120, 180)
        border_color = (150, 200, 255)
    else:
        button_color = (60, 80, 120)
        border_color = (100, 150, 200)
    pygame.draw.rect(surface, button_color, gui.collapse_button_rect, border_radius=4)
    pygame.draw.rect(surface, border_color, gui.collapse_button_rect, 2, border_radius=4)
    if getattr(gui, "control_panel_animating", False):
        button_text = ">" if gui.control_panel_target_collapsed else "<"
    else:
        button_text = "<" if not gui.control_panel_collapsed else ">"
    button_surf = gui.font.render(button_text, True, (200, 220, 255))
    button_rect = button_surf.get_rect(center=gui.collapse_button_rect.center)
    surface.blit(button_surf, button_rect)

    try:
        for widget in gui.widget_manager.widgets:
            if isinstance(widget, dropdown_widget_cls) and getattr(widget, "is_open", False):
                try:
                    scroll_offset = gui.control_panel_scroll if getattr(gui, "control_panel_can_scroll", False) else 0
                    widget.render_menu(
                        surface,
                        alpha=content_alpha,
                        scroll_offset=scroll_offset,
                        panel_rect=gui.control_panel_rect,
                    )
                except Exception as e:
                    logger.warning("Dropdown menu render failed: %s", e)
    except Exception:
        pass

    gui._render_tooltips(surface, mouse_pos)
