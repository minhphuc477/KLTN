"""Helpers for minimap rendering and click interactions."""

import math
import time
from typing import Any, Tuple

from src.core.definitions import SEMANTIC_PALETTE


def render_minimap(gui: Any, pygame: Any) -> None:
    """Render small dungeon overview map in bottom-right corner."""
    if not getattr(gui, "env", None):
        return

    minimap_margin = 20
    minimap_x = gui.screen_w - gui.SIDEBAR_WIDTH - gui.minimap_size - minimap_margin
    minimap_y = gui.screen_h - gui.HUD_HEIGHT - gui.minimap_size - minimap_margin

    minimap = pygame.Surface((gui.minimap_size, gui.minimap_size), pygame.SRCALPHA)
    pygame.draw.rect(minimap, (40, 40, 60, 220), minimap.get_rect(), border_radius=8)

    title_font = pygame.font.SysFont("Arial", 10, bold=True)
    title_surf = title_font.render("Dungeon Map", True, (180, 180, 200))
    minimap.blit(title_surf, (5, 3))

    map_h, map_w = gui.env.height, gui.env.width
    content_area = gui.minimap_size - 30
    scale_x = content_area / map_w
    scale_y = content_area / map_h
    scale = min(scale_x, scale_y)

    scaled_w = int(map_w * scale)
    scaled_h = int(map_h * scale)
    offset_x = (gui.minimap_size - scaled_w) // 2
    offset_y = 18 + (gui.minimap_size - 18 - scaled_h) // 2

    for r in range(map_h):
        for c in range(map_w):
            tile_id = gui.env.grid[r, c]
            if tile_id == SEMANTIC_PALETTE["VOID"]:
                continue
            if tile_id == SEMANTIC_PALETTE["WALL"] or tile_id == SEMANTIC_PALETTE["BLOCK"]:
                color = (60, 60, 80)
            elif tile_id == SEMANTIC_PALETTE["START"]:
                color = (80, 180, 80)
            elif tile_id == SEMANTIC_PALETTE["TRIFORCE"]:
                color = (255, 215, 0)
            elif tile_id in [SEMANTIC_PALETTE["KEY_SMALL"], SEMANTIC_PALETTE["KEY_BOSS"]]:
                color = (255, 200, 50)
            elif tile_id in [
                SEMANTIC_PALETTE["DOOR_LOCKED"],
                SEMANTIC_PALETTE["DOOR_BOMB"],
                SEMANTIC_PALETTE["DOOR_BOSS"],
            ]:
                color = (180, 100, 50)
            elif tile_id == SEMANTIC_PALETTE["STAIR"]:
                color = (100, 150, 255)
            elif tile_id == SEMANTIC_PALETTE["ENEMY"]:
                color = (200, 50, 50)
            else:
                color = (100, 120, 140)

            mini_x = offset_x + int(c * scale)
            mini_y = offset_y + int(r * scale)
            mini_w = max(1, int(scale))
            mini_h = max(1, int(scale))
            pygame.draw.rect(minimap, color, (mini_x, mini_y, mini_w, mini_h))

    pr, pc = gui.env.state.position
    player_x = offset_x + int(pc * scale)
    player_y = offset_y + int(pr * scale)
    player_size = max(2, int(scale * 1.5))
    pygame.draw.circle(minimap, (255, 100, 100), (player_x, player_y), player_size)
    pygame.draw.circle(minimap, (255, 255, 255), (player_x, player_y), player_size + 1, 1)

    current_time = time.time()
    pulse = (math.sin(current_time * 3) + 1) / 2

    for pos in gui.env._find_all_positions(SEMANTIC_PALETTE["KEY_SMALL"]):
        if pos not in gui.env.state.collected_items:
            r, c = pos
            mini_x = offset_x + int(c * scale)
            mini_y = offset_y + int(r * scale)
            size = int(2 + pulse * 2)
            pygame.draw.circle(minimap, (255, 255, 0), (mini_x, mini_y), size)

    for pos in gui.env._find_all_positions(SEMANTIC_PALETTE["KEY_BOSS"]):
        if pos not in gui.env.state.collected_items:
            r, c = pos
            mini_x = offset_x + int(c * scale)
            mini_y = offset_y + int(r * scale)
            size = int(2 + pulse * 2)
            pygame.draw.circle(minimap, (255, 150, 0), (mini_x, mini_y), size)

    pygame.draw.rect(minimap, (70, 70, 100), minimap.get_rect(), 2, border_radius=8)
    gui.screen.blit(minimap, (minimap_x, minimap_y))


def handle_minimap_click(gui: Any, mouse_pos: Tuple[int, int]) -> bool:
    """Handle click on minimap and recenter view to selected tile."""
    if not getattr(gui, "show_minimap", False) or not getattr(gui, "env", None):
        return False

    minimap_margin = 20
    minimap_x = gui.screen_w - gui.SIDEBAR_WIDTH - gui.minimap_size - minimap_margin
    minimap_y = gui.screen_h - gui.HUD_HEIGHT - gui.minimap_size - minimap_margin

    mx, my = mouse_pos
    if not (
        minimap_x <= mx <= minimap_x + gui.minimap_size
        and minimap_y <= my <= minimap_y + gui.minimap_size
    ):
        return False

    map_h, map_w = gui.env.height, gui.env.width
    content_area = gui.minimap_size - 30
    scale_x = content_area / map_w
    scale_y = content_area / map_h
    scale = min(scale_x, scale_y)

    scaled_w = int(map_w * scale)
    scaled_h = int(map_h * scale)
    offset_x = (gui.minimap_size - scaled_w) // 2
    offset_y = 18 + (gui.minimap_size - 18 - scaled_h) // 2

    local_x = mx - minimap_x - offset_x
    local_y = my - minimap_y - offset_y

    if local_x < 0 or local_y < 0:
        return True

    tile_c = int(local_x / scale)
    tile_r = int(local_y / scale)

    if 0 <= tile_r < map_h and 0 <= tile_c < map_w:
        gui.view_offset_x = int(tile_c * gui.TILE_SIZE - (gui.screen_w - gui.SIDEBAR_WIDTH) / 2)
        gui.view_offset_y = int(tile_r * gui.TILE_SIZE - (gui.screen_h - gui.HUD_HEIGHT) / 2)
        gui._clamp_view_offset()
        gui.message = f"Jumped to ({tile_r}, {tile_c})"

    return True
