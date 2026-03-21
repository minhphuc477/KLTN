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

    if widget.flag_name == "show_heatmap" and old_value != widget.checked:
        gui.show_heatmap = widget.checked
        if gui.renderer:
            gui.renderer.show_heatmap = widget.checked
        gui._set_message(f"Heatmap: {'ON' if widget.checked else 'OFF'}")
    elif widget.flag_name == "show_path" and old_value != widget.checked:
        gui._set_message(f"Path overlay: {'ON' if widget.checked else 'OFF'}", 1.5)
    elif widget.flag_name == "show_minimap":
        gui.show_minimap = widget.checked
        gui._set_message(f"Minimap: {'ON' if widget.checked else 'OFF'}")
    elif widget.flag_name == "show_topology" and old_value != widget.checked:
        gui.show_topology = widget.checked
        if widget.checked:
            current = gui.maps[gui.current_map_idx]
            if not hasattr(current, "graph") or not current.graph:
                gui._set_message("Topology not available for this map", 3.0)
            else:
                gui._set_message("Topology overlay: ON", 2.0)
        else:
            gui._set_message("Topology overlay: OFF", 1.2)
    elif widget.flag_name == "show_topology_legend" and old_value != widget.checked:
        gui.show_topology_legend = widget.checked
        gui._set_message(f"Topology legend: {'ON' if widget.checked else 'OFF'}", 1.8)


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
    elif widget.control_name == "difficulty":
        gui.difficulty_idx = widget.selected
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
            gui.ara_weight = float(selected_val)
            gui._set_message(f"ARA* weight: {gui.ara_weight:g}", 1.2)
        except Exception:
            gui.ara_weight = 1.0
    elif widget.control_name == "presets":
        old = gui.current_preset_idx
        gui.current_preset_idx = widget.selected
        if old != gui.current_preset_idx:
            preset_name = gui.presets[gui.current_preset_idx]
            apply_preset_feature_flags(gui.feature_flags, preset_name)
            gui._set_message(f"Preset applied: {preset_name}")


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
            except Exception as exc:
                logger.warning("DROPDOWN: Failed to terminate solver process: %s", exc)
        if hasattr(gui, "preview_thread") and gui.preview_thread:
            gui.preview_thread = None
        if hasattr(gui, "preview_proc") and gui.preview_proc:
            try:
                gui.preview_proc.terminate()
            except Exception:
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
