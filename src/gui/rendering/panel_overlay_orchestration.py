"""Panel, debug, and help overlay orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def render_debug_overlay(*, gui, surface, pygame, time_module, render_debug_overlay_helper):
    render_debug_overlay_helper(gui, surface, pygame, time_module)


def render_unified_bottom_panel(*, gui, pygame, render_unified_bottom_panel_helper):
    render_unified_bottom_panel_helper(gui, pygame)


def render_message_section(*, gui, x, y, width, height, render_message_section_helper):
    render_message_section_helper(gui, x, y, width, height)


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
    render_progress_bar_helper,
):
    render_progress_bar_helper(
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


def render_inventory_section(*, gui, x, y, width, height, pygame, time_module, logger, render_inventory_section_helper):
    render_inventory_section_helper(gui, x, y, width, height, pygame, time_module, logger)


def render_metrics_section(*, gui, x, y, width, height, render_metrics_section_helper):
    render_metrics_section_helper(gui, x, y, width, height)


def render_controls_section(*, gui, x, y, width, height, render_controls_section_helper):
    render_controls_section_helper(gui, x, y, width, height)


def render_status_section(*, gui, x, y, width, height, render_status_section_helper):
    render_status_section_helper(gui, x, y, width, height)


def render_help_overlay(*, gui, pygame, render_help_overlay_helper):
    render_help_overlay_helper(gui, pygame)
