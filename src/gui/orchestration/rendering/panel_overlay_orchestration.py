"""Panel, debug, and help overlay orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.rendering.bottom_panel import (
    render_controls_section as _render_controls_section_helper,
    render_inventory_section as _render_inventory_section_helper,
    render_message_section as _render_message_section_helper,
    render_metrics_section as _render_metrics_section_helper,
    render_progress_bar as _render_progress_bar_helper,
    render_status_section as _render_status_section_helper,
    render_unified_bottom_panel as _render_unified_bottom_panel_helper,
)
from src.gui.rendering.debug_overlay import (
    render_debug_overlay as _render_debug_overlay_helper,
)
from src.gui.rendering.help_overlay import (
    render_help_overlay as _render_help_overlay_helper,
)


def render_debug_overlay(*, gui, surface, pygame, time_module):
    _render_debug_overlay_helper(gui, surface, pygame, time_module)


def render_unified_bottom_panel(*, gui, pygame):
    _render_unified_bottom_panel_helper(gui, pygame)


def render_message_section(*, gui, x, y, width, height):
    _render_message_section_helper(gui, x, y, width, height)


def render_progress_bar(
    *,
    surface,
    x,
    y,
    width,
    height,
    filled,
    total,
    color_filled,
    color_empty,
    pygame,
):
    _render_progress_bar_helper(
        surface,
        x,
        y,
        width,
        height,
        filled,
        total,
        color_filled,
        color_empty,
        pygame,
    )


def render_inventory_section(*, gui, x, y, width, height, pygame, time_module, logger):
    _render_inventory_section_helper(gui, x, y, width, height, pygame, time_module, logger)


def render_metrics_section(*, gui, x, y, width, height):
    _render_metrics_section_helper(gui, x, y, width, height)


def render_controls_section(*, gui, x, y, width, height):
    _render_controls_section_helper(gui, x, y, width, height)


def render_status_section(*, gui, x, y, width, height):
    _render_status_section_helper(gui, x, y, width, height)


def render_help_overlay(*, gui, pygame):
    _render_help_overlay_helper(gui, pygame)
