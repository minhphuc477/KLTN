"""Helpers for map navigation and viewport movement."""

from typing import Any


def next_map(gui: Any, logger: Any) -> None:
    """Move to next map and stop auto-solve if active."""
    try:
        if getattr(gui, "auto_mode", False):
            gui._stop_auto("map change (next)")
    except Exception:
        pass
    try:
        gui.current_map_idx = (gui.current_map_idx + 1) % len(gui.maps)
        gui._load_current_map()
        gui._center_view()
        if getattr(gui, "effects", None):
            gui.effects.clear()
        gui.step_count = 0
    except Exception:
        logger.exception("_next_map failed")


def prev_map(gui: Any, logger: Any) -> None:
    """Move to previous map and stop auto-solve if active."""
    try:
        if getattr(gui, "auto_mode", False):
            gui._stop_auto("map change (prev)")
    except Exception:
        pass
    try:
        gui.current_map_idx = (gui.current_map_idx - 1) % len(gui.maps)
        gui._load_current_map()
        gui._center_view()
        if getattr(gui, "effects", None):
            gui.effects.clear()
        gui.step_count = 0
    except Exception:
        logger.exception("_prev_map failed")


def clamp_view_offset(gui: Any) -> None:
    """Clamp view offset to valid map bounds while allowing small-map negative panning."""
    if getattr(gui, "env", None) is None:
        return

    map_w = gui.env.width * gui.TILE_SIZE
    map_h = gui.env.height * gui.TILE_SIZE
    view_w = gui.screen_w - gui.SIDEBAR_WIDTH
    view_h = gui.screen_h - gui.HUD_HEIGHT

    min_offset_x = min(0, map_w - view_w)
    max_offset_x = max(0, map_w - view_w)
    min_offset_y = min(0, map_h - view_h)
    max_offset_y = max(0, map_h - view_h)

    gui.view_offset_x = min(max(gui.view_offset_x, min_offset_x), max_offset_x)
    gui.view_offset_y = min(max(gui.view_offset_y, min_offset_y), max_offset_y)


def center_on_player(gui: Any) -> None:
    """Center viewport on current player position then clamp."""
    if getattr(gui, "env", None) is None:
        return
    r, c = gui.env.state.position
    player_x = c * gui.TILE_SIZE
    player_y = r * gui.TILE_SIZE

    view_w = gui.screen_w - gui.SIDEBAR_WIDTH
    view_h = gui.screen_h - gui.HUD_HEIGHT

    gui.view_offset_x = player_x - view_w // 2
    gui.view_offset_y = player_y - view_h // 2
    gui._clamp_view_offset()
