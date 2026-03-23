"""Control-panel animation orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.control_panel.animation import (
    start_toggle_panel_animation as _start_toggle_panel_animation_helper,
    update_control_panel_animation as _update_control_panel_animation_helper,
)
from src.gui.control_panel.scroll import (
    update_control_panel_scroll as _update_control_panel_scroll_helper,
)


def start_toggle_panel_animation(*, gui, target_collapsed, time_module):
    _start_toggle_panel_animation_helper(gui, target_collapsed, time_module)


def update_control_panel_animation(*, gui, time_module):
    _update_control_panel_animation_helper(gui, time_module)


def update_control_panel_scroll(*, gui, time_module):
    _update_control_panel_scroll_helper(gui, time_module)
