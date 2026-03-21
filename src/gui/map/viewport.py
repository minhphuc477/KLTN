"""Viewport and zoom control helpers for the GUI."""

from typing import Any


def center_view(gui: Any) -> None:
    """Center current map in view and clamp offsets."""
    if gui.env is None:
        return
    map_w = gui.env.width * gui.TILE_SIZE
    map_h = gui.env.height * gui.TILE_SIZE
    view_w = gui.screen_w - gui.SIDEBAR_WIDTH
    view_h = gui.screen_h - gui.HUD_HEIGHT
    gui.view_offset_x = (map_w - view_w) // 2
    gui.view_offset_y = (map_h - view_h) // 2
    gui._clamp_view_offset()


def auto_fit_zoom(gui: Any) -> None:
    """Set largest zoom level that fits current map in view."""
    if gui.env is None:
        return

    view_w = gui.screen_w - gui.SIDEBAR_WIDTH - 20
    view_h = gui.screen_h - gui.HUD_HEIGHT - 20
    map_h = gui.env.height
    map_w = gui.env.width

    best_zoom_idx = 0
    for idx, tile_size in enumerate(gui.ZOOM_LEVELS):
        if map_w * tile_size <= view_w and map_h * tile_size <= view_h:
            best_zoom_idx = idx
        else:
            break

    if best_zoom_idx != gui.zoom_idx:
        gui.zoom_idx = best_zoom_idx
        gui.TILE_SIZE = gui.ZOOM_LEVELS[gui.zoom_idx]
        gui._load_assets()
        if gui.renderer:
            gui.renderer.set_tile_size(gui.TILE_SIZE)

    center_view(gui)
    if gui.renderer and gui.env and gui.env.start_pos:
        gui.renderer.set_agent_position(gui.env.start_pos[0], gui.env.start_pos[1], immediate=True)


def change_zoom(gui: Any, delta: int, center: tuple | None = None) -> None:
    """Change zoom and keep selected screen position stable in map space."""
    old_idx = gui.zoom_idx
    new_idx = max(0, min(len(gui.ZOOM_LEVELS) - 1, gui.zoom_idx + delta))
    if new_idx == old_idx:
        return

    if center is None:
        center_x = (gui.screen_w - gui.SIDEBAR_WIDTH) // 2
        center_y = (gui.screen_h - gui.HUD_HEIGHT) // 2
    else:
        center_x, center_y = center
        center_x = max(0, min(center_x, gui.screen_w - gui.SIDEBAR_WIDTH))
        center_y = max(0, min(center_y, gui.screen_h - gui.HUD_HEIGHT))

    old_tile = gui.TILE_SIZE
    tile_x = (gui.view_offset_x + center_x) / float(old_tile) if old_tile else 0.0
    tile_y = (gui.view_offset_y + center_y) / float(old_tile) if old_tile else 0.0

    gui.zoom_idx = new_idx
    gui.TILE_SIZE = gui.ZOOM_LEVELS[gui.zoom_idx]
    gui._load_assets()
    if gui.renderer:
        gui.renderer.set_tile_size(gui.TILE_SIZE)

    gui.view_offset_x = int(tile_x * gui.TILE_SIZE - center_x)
    gui.view_offset_y = int(tile_y * gui.TILE_SIZE - center_y)
    gui._clamp_view_offset()

    gui.message = f"Zoom: {gui.TILE_SIZE}px"
