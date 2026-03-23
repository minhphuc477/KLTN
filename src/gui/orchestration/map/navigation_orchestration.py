"""Map navigation/view/minimap orchestration bridges for ZeldaGUI."""

from __future__ import annotations

import os

try:
    import pygame
except ImportError:  # pragma: no cover - optional in headless tests
    pygame = None

from src.gui.map.minimap import handle_minimap_click as _handle_minimap_click, render_minimap as _render_minimap
from src.gui.map.navigation import (
    center_on_player as _center_on_player,
    clamp_view_offset as _clamp_view_offset,
    next_map as _next_map,
    prev_map as _prev_map,
)
from src.gui.map.viewport import auto_fit_zoom as _auto_fit_zoom, center_view as _center_view, change_zoom as _change_zoom


def load_current_map(
    *,
    gui,
    logger,
    zelda_logic_env_cls,
    sanity_checker_cls,
    semantic_palette,
    load_current_map_helper,
):
    load_current_map_helper(
        gui,
        os_module=os,
        logger=logger,
        zelda_logic_env_cls=zelda_logic_env_cls,
        sanity_checker_cls=sanity_checker_cls,
        semantic_palette=semantic_palette,
    )


def center_view(*, gui):
    _center_view(gui)


def auto_fit_zoom(*, gui):
    _auto_fit_zoom(gui)


def change_zoom(*, gui, delta, center):
    _change_zoom(gui, delta, center)


def next_map(*, gui, logger):
    _next_map(gui, logger)


def prev_map(*, gui, logger):
    _prev_map(gui, logger)


def clamp_view_offset(*, gui):
    _clamp_view_offset(gui)


def center_on_player(*, gui):
    _center_on_player(gui)


def render_minimap(*, gui):
    _render_minimap(gui, pygame)


def handle_minimap_click(*, gui, mouse_pos):
    return _handle_minimap_click(gui, mouse_pos)
