"""Map navigation/view/minimap orchestration bridges for ZeldaGUI."""

from __future__ import annotations


def load_current_map(
    *,
    gui,
    os_module,
    logger,
    zelda_logic_env_cls,
    sanity_checker_cls,
    semantic_palette,
    load_current_map_helper,
):
    load_current_map_helper(
        gui,
        os_module=os_module,
        logger=logger,
        zelda_logic_env_cls=zelda_logic_env_cls,
        sanity_checker_cls=sanity_checker_cls,
        semantic_palette=semantic_palette,
    )


def center_view(*, gui, center_view_helper):
    center_view_helper(gui)


def auto_fit_zoom(*, gui, auto_fit_zoom_helper):
    auto_fit_zoom_helper(gui)


def change_zoom(*, gui, delta, center, change_zoom_helper):
    change_zoom_helper(gui, delta, center)


def next_map(*, gui, logger, next_map_helper):
    next_map_helper(gui, logger)


def prev_map(*, gui, logger, prev_map_helper):
    prev_map_helper(gui, logger)


def clamp_view_offset(*, gui, clamp_view_offset_helper):
    clamp_view_offset_helper(gui)


def center_on_player(*, gui, center_on_player_helper):
    center_on_player_helper(gui)


def render_minimap(*, gui, pygame, render_minimap_helper):
    render_minimap_helper(gui, pygame)


def handle_minimap_click(*, gui, mouse_pos, handle_minimap_click_helper):
    return handle_minimap_click_helper(gui, mouse_pos)
