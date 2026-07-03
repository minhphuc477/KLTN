"""Helpers for applying control-panel widget state updates to GUI objects."""

from typing import Any

from src.gui.controls.control_panel_logic import (
    algorithm_label,
    apply_preset_feature_flags,
    difficulty_label,
    representation_from_dropdown,
    zoom_label,
    zoom_level_index_from_dropdown,
)


def apply_control_panel_widget_updates(gui: Any, widget_manager: Any, checkbox_type: Any, logger: Any) -> None:
    """Apply checkbox and dropdown state updates for all widgets."""
    for widget in widget_manager.widgets:
        if isinstance(widget, checkbox_type) and hasattr(widget, "flag_name"):
            apply_checkbox_widget_update(gui, widget, logger)
        apply_dropdown_widget_update(gui, widget, logger)


def apply_checkbox_widget_update(gui: Any, widget: Any, logger: Any) -> None:
    """Apply one checkbox widget state update to GUI fields."""
    old_value = gui.feature_flags.get(widget.flag_name, False)
    gui.feature_flags[widget.flag_name] = widget.checked
    logger.info("Feature flag set: %s=%s", widget.flag_name, widget.checked)
    changed = old_value != widget.checked

    if widget.flag_name == "show_heatmap" and changed:
        gui.show_heatmap = widget.checked
        if gui.renderer:
            gui.renderer.show_heatmap = widget.checked
        gui._set_message(f"Heatmap: {'ON' if widget.checked else 'OFF'}")
    elif widget.flag_name == "show_path" and changed:
        gui._set_message(f"Path overlay: {'ON' if widget.checked else 'OFF'}", 1.5)
    elif widget.flag_name == "show_minimap" and (
        changed or bool(getattr(gui, "show_minimap", False)) != bool(widget.checked)
    ):
        gui.show_minimap = widget.checked
        gui._set_message(f"Minimap: {'ON' if widget.checked else 'OFF'}")
    elif widget.flag_name == "show_topology" and changed:
        gui.show_topology = widget.checked
        if widget.checked:
            current = gui.maps[gui.current_map_idx]
            if not hasattr(current, "graph") or not current.graph:
                gui._set_message("Topology overlay: ON (inferred from stitched grid)", 2.5)
            else:
                gui._set_message("Topology overlay: ON", 2.0)
        else:
            gui._set_message("Topology overlay: OFF", 1.2)
    elif widget.flag_name == "show_topology_legend" and old_value != widget.checked:
        gui.show_topology_legend = widget.checked
        gui._set_message(f"Topology legend: {'ON' if widget.checked else 'OFF'}", 1.8)
    elif widget.flag_name == "force_grid" and changed:
        gui.force_grid_algorithm = bool(widget.checked)
        gui._set_message(f"Force grid solver: {'ON' if widget.checked else 'OFF'}", 1.5)


def apply_dropdown_widget_update(gui: Any, widget: Any, logger: Any) -> None:
    """Apply one dropdown widget state update to GUI fields."""
    if not hasattr(widget, "control_name"):
        return

    if widget.control_name == "zoom":
        old_zoom_idx = gui.zoom_level_idx
        gui.zoom_level_idx = widget.selected
        if old_zoom_idx != gui.zoom_level_idx:
            new_zoom_idx = zoom_level_index_from_dropdown(gui.zoom_level_idx)
            if new_zoom_idx is not None and new_zoom_idx != gui.zoom_idx:
                gui.zoom_idx = new_zoom_idx
                gui.TILE_SIZE = gui.ZOOM_LEVELS[gui.zoom_idx]
                gui._load_assets()
                gui._center_view()
                gui.message = f"Zoom: {zoom_label(gui.zoom_level_idx)}"
    elif widget.control_name == "level":
        apply_level_dropdown_update(gui, widget, logger)
    elif widget.control_name == "floor":
        old_floor = int(getattr(gui, "current_floor", 1))
        gui.current_floor = int(getattr(widget, "selected", 0)) + 1
        if old_floor != gui.current_floor:
            gui._set_message(f"Floor: {gui.current_floor}", 1.2)
    elif widget.control_name == "difficulty":
        old_difficulty = gui.difficulty_idx
        gui.difficulty_idx = widget.selected
        if old_difficulty != gui.difficulty_idx:
            gui.message = f"Difficulty: {difficulty_label(gui.difficulty_idx)}"
    elif widget.control_name == "algorithm":
        apply_algorithm_dropdown_update(gui, widget, logger)
    elif widget.control_name == "representation":
        old_rep = getattr(gui, "search_representation", "hybrid")
        gui.search_representation = representation_from_dropdown(widget.selected, old_rep)
        if old_rep != gui.search_representation:
            gui._set_message(f"Search space: {gui.search_representation}")
    elif widget.control_name == "ara_weight":
        try:
            selected_val = widget.options[widget.selected]
            old_weight = float(getattr(gui, "ara_weight", 1.0))
            gui.ara_weight = float(selected_val)
            if old_weight != gui.ara_weight:
                gui._set_message(f"Weighted A* weight: {gui.ara_weight:g}", 1.2)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            gui.ara_weight = 1.0
    elif widget.control_name == "presets":
        old = gui.current_preset_idx
        gui.current_preset_idx = widget.selected
        if old != gui.current_preset_idx:
            preset_name = gui.presets[gui.current_preset_idx]
            apply_preset_feature_flags(gui.feature_flags, preset_name)
            gui._set_message(f"Preset applied: {preset_name}")
    elif widget.control_name == "match_threshold":
        try:
            selected_val = widget.options[widget.selected]
            old_threshold = float(getattr(gui, "match_apply_threshold", 0.85))
            gui.match_apply_threshold = float(selected_val)
            if old_threshold != gui.match_apply_threshold:
                gui._set_message(f"Match threshold: {gui.match_apply_threshold:.2f}", 1.2)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            gui.match_apply_threshold = 0.85


def apply_level_dropdown_update(gui: Any, widget: Any, logger: Any) -> None:
    """Switch to the selected loaded/generated level and reset solve state."""
    maps = list(getattr(gui, "maps", []) or [])
    if not maps:
        return

    selected = max(0, min(int(getattr(widget, "selected", 0)), len(maps) - 1))
    old_idx = int(getattr(gui, "current_map_idx", 0))
    if selected == old_idx:
        return

    if getattr(gui, "solver_proc", None):
        try:
            gui.solver_proc.terminate()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("LEVEL: Failed to terminate solver process: %s", exc)
    if getattr(gui, "preview_proc", None):
        try:
            gui.preview_proc.terminate()
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass
    if hasattr(gui, "preview_thread"):
        gui.preview_thread = None

    if hasattr(gui, "_clear_solver_state"):
        gui._clear_solver_state(reason="level changed")

    gui.current_map_idx = selected
    gui._load_current_map()
    gui._center_view()

    if getattr(gui, "effects", None):
        try:
            gui.effects.clear()
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass
    gui.step_count = 0
    gui.auto_path = []
    gui.auto_mode = False

    names = list(getattr(gui, "map_names", []) or [])
    level_name = names[selected] if selected < len(names) else f"Level {selected + 1}"
    gui._set_message(f"Loaded {level_name} (press SPACE or Solve Level)", 2.5)


def apply_algorithm_dropdown_update(gui: Any, widget: Any, logger: Any) -> None:
    """Apply algorithm dropdown update and handle solver-state transitions."""
    old_algorithm_idx = gui.algorithm_idx
    gui.algorithm_idx = widget.selected
    if old_algorithm_idx == gui.algorithm_idx:
        return

    old_algorithm_name = algorithm_label(old_algorithm_idx)
    new_algorithm_name = algorithm_label(gui.algorithm_idx)
    gui.message = f"Solver: {new_algorithm_name}"
    logger.info(
        "DROPDOWN: Algorithm changed from %d(%s) to %d(%s)",
        old_algorithm_idx,
        old_algorithm_name,
        gui.algorithm_idx,
        new_algorithm_name,
    )

    if getattr(gui, "solver_running", False):
        logger.info("DROPDOWN: Stopping solver running with old algorithm %s", old_algorithm_name)
        if hasattr(gui, "solver_proc") and gui.solver_proc:
            try:
                gui.solver_proc.terminate()
                logger.info("DROPDOWN: Terminated solver process")
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.warning("DROPDOWN: Failed to terminate solver process: %s", exc)
        if hasattr(gui, "preview_thread") and gui.preview_thread:
            gui.preview_thread = None
        if hasattr(gui, "preview_proc") and gui.preview_proc:
            try:
                gui.preview_proc.terminate()
            except (AttributeError, RuntimeError, ValueError, TypeError):
                pass
            gui.preview_proc = None
        gui._clear_solver_state(reason=f"algorithm changed to {new_algorithm_name}")
        gui._set_message(f"Switched to {new_algorithm_name} (press SPACE to solve)", 2.5)
        return

    if bool(gui.auto_path):
        logger.info("ALGORITHM CHANGED: %s -> %s", old_algorithm_name, new_algorithm_name)
        logger.info("Triggering automatic resolve to show new path")
        gui.auto_path = []
        gui.auto_mode = False
        gui._set_message(f"Recomputing with {new_algorithm_name}...", 2.0)
        gui._pending_solver_trigger = True

