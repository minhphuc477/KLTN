"""Asset and control-panel boot orchestration bridges for ZeldaGUI."""

from __future__ import annotations

from src.gui.rendering.link_sprite_builder import (
    build_link_sprite as _build_link_sprite_helper,
)
from src.gui.rendering.tile_asset_builder import (
    build_stair_marker_sprite as _build_stair_marker_sprite_helper,
    build_tile_images as _build_tile_images_helper,
    default_tile_color_map as _default_tile_color_map_helper,
)


def load_assets(*, gui, semantic_palette, pygame):
    """Build base tile assets, link sprite, and optional stair marker sprite."""
    color_map = _default_tile_color_map_helper(semantic_palette=semantic_palette)
    gui.images = _build_tile_images_helper(
        tile_size=gui.TILE_SIZE,
        color_map=color_map,
        semantic_palette=semantic_palette,
        pygame=pygame,
    )

    gui.link_img = gui._create_link_sprite()

    try:
        gui.stair_sprite, gui.stair_anim_phase = _build_stair_marker_sprite_helper(
            tile_size=gui.TILE_SIZE,
            pygame=pygame,
        )
    except Exception as exc:
        gui.stair_sprite = None
        gui.stair_anim_phase = 0.0


def create_link_sprite(*, tile_size, pygame):
    """Create the link sprite surface for the current tile size."""
    return _build_link_sprite_helper(tile_size=tile_size, pygame=pygame)


def init_control_panel(*, gui, widgets_available, widget_manager_cls):
    """Initialize widget manager and compute panel layout when widgets are available."""
    if not widgets_available:
        return

    gui.widget_manager = widget_manager_cls()
    gui._update_control_panel_positions()
