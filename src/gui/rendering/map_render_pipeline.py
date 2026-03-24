"""Map-layer render pipeline helpers extracted from gui_runner._render."""

from __future__ import annotations

from typing import Any, Tuple


def create_map_surface(*, gui: Any, pygame: Any) -> Tuple[Any, int, int]:
    """Create and initialize the map surface for the current frame."""
    view_w = max(1, int(gui.screen_w - gui.SIDEBAR_WIDTH))
    view_h = max(1, int(gui.screen_h - gui.HUD_HEIGHT))

    try:
        map_surface = pygame.Surface((view_w, view_h)).convert()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        map_surface = pygame.Surface((view_w, view_h))
    map_surface.fill((20, 20, 30))
    return map_surface, view_w, view_h


def compute_visible_bounds(*, gui: Any, view_w: int, view_h: int, map_h: int, map_w: int) -> Tuple[int, int, int, int]:
    """Compute inclusive-exclusive row/col ranges for visible map cells."""
    start_c = max(0, int(gui.view_offset_x) // gui.TILE_SIZE)
    start_r = max(0, int(gui.view_offset_y) // gui.TILE_SIZE)
    end_c = min(map_w, start_c + (view_w // gui.TILE_SIZE) + 2)
    end_r = min(map_h, start_r + (view_h // gui.TILE_SIZE) + 2)
    return start_r, end_r, start_c, end_c


def collect_item_render_state(*, gui: Any, semantic_palette: dict) -> Tuple[set, Tuple[int, ...]]:
    """Build collected-item and collectible-id state used by tile rendering."""
    env_collected = getattr(gui.env.state, "collected_items", set()) or set()
    gui_collected = getattr(gui, "collected_positions", set()) or set()
    collected_items = env_collected | gui_collected

    collectible_tile_ids = (
        semantic_palette.get("KEY_SMALL", -1),
        semantic_palette.get("KEY_BOSS", -1),
        semantic_palette.get("ITEM_BOMB", -1),
        semantic_palette.get("KEY_ITEM", -1),
        semantic_palette.get("ITEM_MINOR", -1),
    )
    return collected_items, collectible_tile_ids


def render_visible_tiles(
    *,
    gui: Any,
    map_surface: Any,
    start_r: int,
    end_r: int,
    start_c: int,
    end_c: int,
    collected_items: set,
    collectible_tile_ids: Tuple[int, ...],
    animating_block_positions: set,
    semantic_palette: dict,
    math_module: Any,
    time_module: Any,
) -> int:
    """Render visible tiles using sprite renderer when available, fallback otherwise."""
    tiles_drawn = 0

    if gui.renderer:
        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                tile_id = gui.env.grid[r, c]
                screen_x = c * gui.TILE_SIZE - gui.view_offset_x
                screen_y = r * gui.TILE_SIZE - gui.view_offset_y

                if (r, c) in collected_items and tile_id in collectible_tile_ids:
                    tile_id = semantic_palette["FLOOR"]

                if (r, c) in animating_block_positions and tile_id == semantic_palette["BLOCK"]:
                    tile_id = semantic_palette["FLOOR"]

                tile_surface = gui.renderer.sprite_manager.get_tile(tile_id, gui.TILE_SIZE)
                map_surface.blit(tile_surface, (screen_x, screen_y))
                tiles_drawn += 1

                if tile_id == semantic_palette["STAIR"] and getattr(gui, "stair_sprite", None):
                    try:
                        alpha = int(140 + 90 * math_module.sin(time_module.time() * 3.0))
                        stair_sprite = gui.stair_sprite.copy()
                        stair_sprite.set_alpha(max(20, alpha))
                        sx = screen_x + (gui.TILE_SIZE - stair_sprite.get_width()) // 2
                        sy = screen_y + (gui.TILE_SIZE - stair_sprite.get_height()) // 2
                        map_surface.blit(stair_sprite, (sx, sy))
                    except (AttributeError, RuntimeError, ValueError, TypeError):
                        pass
        return tiles_drawn

    for r in range(start_r, end_r):
        for c in range(start_c, end_c):
            tile_id = gui.env.grid[r, c]

            if (r, c) in collected_items and tile_id in collectible_tile_ids:
                tile_id = semantic_palette["FLOOR"]

            if (r, c) in animating_block_positions and tile_id == semantic_palette["BLOCK"]:
                tile_id = semantic_palette["FLOOR"]

            tile_surface = gui.images.get(tile_id, gui.images.get(semantic_palette["FLOOR"]))
            screen_x = c * gui.TILE_SIZE - gui.view_offset_x
            screen_y = r * gui.TILE_SIZE - gui.view_offset_y
            map_surface.blit(tile_surface, (screen_x, screen_y))
            tiles_drawn += 1

            if tile_id == semantic_palette["STAIR"] and getattr(gui, "stair_sprite", None):
                try:
                    alpha = int(140 + 90 * math_module.sin(time_module.time() * 3.0))
                    stair_sprite = gui.stair_sprite.copy()
                    stair_sprite.set_alpha(max(20, alpha))
                    sx = screen_x + (gui.TILE_SIZE - stair_sprite.get_width()) // 2
                    sy = screen_y + (gui.TILE_SIZE - stair_sprite.get_height()) // 2
                    map_surface.blit(stair_sprite, (sx, sy))
                except (AttributeError, RuntimeError, ValueError, TypeError):
                    pass

    return tiles_drawn


def render_heatmap_overlay(
    *,
    gui: Any,
    map_surface: Any,
    start_r: int,
    end_r: int,
    start_c: int,
    end_c: int,
    pygame: Any,
) -> None:
    """Render search heatmap overlay when enabled."""
    if not gui.show_heatmap or not gui.search_heatmap:
        return

    max_visits = max(gui.search_heatmap.values()) if gui.search_heatmap else 1
    for (r, c), visits in gui.search_heatmap.items():
        if not (start_r <= r < end_r and start_c <= c < end_c):
            continue

        intensity = visits / max_visits
        red = int(255 * intensity)
        blue = int(255 * (1 - intensity))
        heat_color = (red, 0, blue, 100)

        heat_surf = pygame.Surface((gui.TILE_SIZE, gui.TILE_SIZE), pygame.SRCALPHA)
        heat_surf.fill(heat_color)
        screen_x = c * gui.TILE_SIZE - gui.view_offset_x
        screen_y = r * gui.TILE_SIZE - gui.view_offset_y
        map_surface.blit(heat_surf, (screen_x, screen_y))

