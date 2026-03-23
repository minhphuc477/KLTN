"""Post-map UI composition helpers for gui_runner._render."""

from __future__ import annotations

from typing import Any


def draw_sidebar_shell(*, gui: Any, pygame: Any) -> int:
    """Draw sidebar panel background and divider, return sidebar x offset."""
    sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH
    pygame.draw.rect(gui.screen, (35, 35, 50), (sidebar_x, 0, gui.SIDEBAR_WIDTH, gui.screen_h))
    pygame.draw.line(gui.screen, (60, 60, 80), (sidebar_x, 0), (sidebar_x, gui.screen_h), 2)
    return sidebar_x


def render_sidebar_content(
    *,
    gui: Any,
    sidebar_x: int,
    map_w: int,
    map_h: int,
    player_row: int,
    player_col: int,
    pygame: Any,
    time_module: Any,
    math_module: Any,
    semantic_palette: dict,
    logger: Any,
    render_sidebar_header_fn: Any,
    render_sidebar_status_fn: Any,
) -> None:
    """Render extracted sidebar sections."""
    y_pos = 10
    y_pos = render_sidebar_header_fn(
        gui=gui,
        screen=gui.screen,
        sidebar_x=sidebar_x,
        y_pos=y_pos,
        map_w=map_w,
        map_h=map_h,
        time_module=time_module,
        math_module=math_module,
        pygame=pygame,
        logger=logger,
    )

    render_sidebar_status_fn(
        gui=gui,
        screen=gui.screen,
        sidebar_x=sidebar_x,
        y_pos=y_pos,
        player_row=player_row,
        player_col=player_col,
        pygame=pygame,
        time_module=time_module,
        math_module=math_module,
        semantic_palette=semantic_palette,
    )


def render_post_map_layers(*, gui: Any, pygame: Any, logger: Any, render_preview_layer_fn: Any) -> None:
    """Render optional non-map layers and dialogs after base map and sidebar."""
    if gui.show_minimap:
        gui._render_minimap()

    if gui.show_help:
        gui._render_help_overlay()

    render_preview_layer_fn(gui=gui, pygame=pygame, logger=logger)


def render_top_ui_layers(*, gui: Any, logger: Any) -> None:
    """Render top-level overlays and HUD elements in existing z-order."""
    if gui.feature_flags.get("show_topology", False):
        gui.show_topology = True

    if getattr(gui, "show_topology", False):
        try:
            logger.debug("Rendering topology overlay")
            gui._render_topology_overlay(gui.screen)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Topology overlay failed: %s", exc)

    if getattr(gui, "show_solver_comparison_overlay", False):
        try:
            gui._render_solver_comparison_overlay(gui.screen)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Solver comparison overlay failed: %s", exc)

    if gui.control_panel_enabled:
        gui._render_control_panel(gui.screen)

    if getattr(gui, "debug_overlay_enabled", False):
        try:
            gui._render_debug_overlay(gui.screen)
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Debug overlay render failed: %s", exc)

    if gui.auto_mode:
        gui._render_item_legend(gui.screen)

    gui._render_error_banner(gui.screen)
    gui._render_solver_status_banner(gui.screen)
    gui._render_toasts(gui.screen)
