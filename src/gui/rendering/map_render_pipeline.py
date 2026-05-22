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


def _render_lock_affordance_overlay(
    *,
    pygame: Any,
    map_surface: Any,
    tile_id: int,
    screen_x: int,
    screen_y: int,
    tile_size: int,
    semantic_palette: dict,
    time_module: Any,
    math_module: Any,
) -> None:
    """Draw high-contrast lock/gate affordances over procedural or loaded sprites."""
    lock_styles = {
        semantic_palette.get("DOOR_LOCKED", -1): ((255, 220, 70), "key"),
        semantic_palette.get("DOOR_BOSS", -1): ((255, 80, 80), "boss"),
        semantic_palette.get("DOOR_BOMB", -1): ((255, 130, 70), "bomb"),
        semantic_palette.get("DOOR_PUZZLE", -1): ((190, 120, 255), "puzzle"),
    }
    if int(tile_id) not in lock_styles:
        return

    color, glyph = lock_styles[int(tile_id)]
    pulse = 0.5 + 0.5 * math_module.sin(time_module.time() * 4.0)
    alpha = int(70 + 65 * pulse)
    size = int(tile_size)
    x = int(screen_x)
    y = int(screen_y)
    pad = max(1, size // 10)
    stroke = max(2, size // 9)

    glow = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, alpha), (0, 0, size, size), max(2, stroke))
    pygame.draw.rect(glow, (*color, max(35, alpha // 2)), (pad, pad, size - 2 * pad, size - 2 * pad), 1)
    map_surface.blit(glow, (x, y))

    cx = x + size // 2
    cy = y + size // 2
    shackle_w = max(8, size // 2)
    shackle_h = max(6, size // 3)
    body_w = max(9, size // 2)
    body_h = max(8, size // 3)
    body_rect = pygame.Rect(cx - body_w // 2, cy - body_h // 4, body_w, body_h)
    arc_rect = pygame.Rect(cx - shackle_w // 2, cy - shackle_h // 2, shackle_w, shackle_h)

    pygame.draw.arc(map_surface, color, arc_rect, 3.14, 6.28, stroke)
    pygame.draw.rect(map_surface, color, body_rect, border_radius=max(1, size // 12))
    pygame.draw.rect(map_surface, (30, 25, 25), body_rect, max(1, stroke // 2), border_radius=max(1, size // 12))

    if glyph == "boss":
        pygame.draw.circle(map_surface, (255, 245, 230), (cx, cy), max(3, size // 7))
        pygame.draw.circle(map_surface, (40, 20, 20), (cx - max(1, size // 14), cy - 1), max(1, size // 24))
        pygame.draw.circle(map_surface, (40, 20, 20), (cx + max(1, size // 14), cy - 1), max(1, size // 24))
    elif glyph == "bomb":
        pygame.draw.line(map_surface, (45, 25, 20), (x + pad * 2, y + pad * 2), (x + size - pad * 2, y + size - pad * 2), stroke)
        pygame.draw.line(map_surface, (45, 25, 20), (x + size - pad * 2, y + pad * 2), (x + pad * 2, y + size - pad * 2), stroke)
    elif glyph == "puzzle":
        pygame.draw.circle(map_surface, (245, 235, 255), (cx, cy), max(2, size // 9), max(1, stroke // 2))
        pygame.draw.line(map_surface, (245, 235, 255), (cx, cy - size // 5), (cx, cy + size // 5), max(1, stroke // 2))
        pygame.draw.line(map_surface, (245, 235, 255), (cx - size // 5, cy), (cx + size // 5, cy), max(1, stroke // 2))
    else:
        pygame.draw.circle(map_surface, (35, 25, 10), (cx, cy), max(2, size // 12))
        pygame.draw.rect(map_surface, (35, 25, 10), (cx - max(1, size // 28), cy, max(2, size // 14), max(4, size // 6)))


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
    pygame: Any = None,
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
                if pygame is not None:
                    _render_lock_affordance_overlay(
                        pygame=pygame,
                        map_surface=map_surface,
                        tile_id=int(tile_id),
                        screen_x=screen_x,
                        screen_y=screen_y,
                        tile_size=gui.TILE_SIZE,
                        semantic_palette=semantic_palette,
                        time_module=time_module,
                        math_module=math_module,
                    )
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
            if pygame is not None:
                _render_lock_affordance_overlay(
                    pygame=pygame,
                    map_surface=map_surface,
                    tile_id=int(tile_id),
                    screen_x=screen_x,
                    screen_y=screen_y,
                    tile_size=gui.TILE_SIZE,
                    semantic_palette=semantic_palette,
                    time_module=time_module,
                    math_module=math_module,
                )
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

