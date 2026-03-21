"""Control-panel animation orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def start_toggle_panel_animation(*, gui, target_collapsed, time_module, start_toggle_panel_animation_helper):
    start_toggle_panel_animation_helper(gui, target_collapsed, time_module)


def update_control_panel_animation(*, gui, time_module, update_control_panel_animation_helper):
    update_control_panel_animation_helper(gui, time_module)


def update_control_panel_scroll(*, gui, time_module, update_control_panel_scroll_helper):
    update_control_panel_scroll_helper(gui, time_module)
