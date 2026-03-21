"""Helpers for block push animation flow and movement checks."""

from typing import Any, Tuple


def start_block_push_animation(gui: Any, block_from: Tuple[int, int], block_to: Tuple[int, int], pygame: Any, logger: Any) -> None:
    """Start animating a block from source to destination."""
    gui.block_push_animations.append(
        {
            "from_pos": block_from,
            "to_pos": block_to,
            "start_time": pygame.time.get_ticks(),
            "duration": gui.block_push_duration,
            "progress": 0.0,
        }
    )
    logger.debug("BLOCK_PUSH: Started animation from %s to %s", block_from, block_to)


def update_block_push_animations(gui: Any, pygame: Any, semantic_palette: dict, pop_effect_cls: Any, logger: Any) -> None:
    """Advance block push animations and commit finished block moves to grid."""
    if not hasattr(gui, "block_push_animations") or not gui.block_push_animations:
        return

    current_time = pygame.time.get_ticks()
    still_active = []

    for anim in gui.block_push_animations:
        elapsed = current_time - anim["start_time"]
        progress = min(1.0, elapsed / anim["duration"])
        anim["progress"] = progress

        if progress < 1.0:
            still_active.append(anim)
            continue

        from_pos = anim["from_pos"]
        to_pos = anim["to_pos"]
        try:
            block_id = semantic_palette["BLOCK"]
            floor_id = semantic_palette["FLOOR"]
            gui.env.grid[from_pos[0], from_pos[1]] = floor_id
            gui.env.grid[to_pos[0], to_pos[1]] = block_id
            logger.debug("BLOCK_PUSH: Animation complete, grid updated: %s->%s", from_pos, to_pos)
            if getattr(gui, "effects", None):
                gui.effects.add_effect(pop_effect_cls(to_pos, (139, 90, 43)))
        except Exception as e:
            logger.warning("BLOCK_PUSH: Failed to update grid after animation: %s", e)

    gui.block_push_animations = still_active


def render_block_push_animations(gui: Any, surface: Any, pygame: Any, semantic_palette: dict) -> None:
    """Render animating blocks with eased interpolation."""
    if not hasattr(gui, "block_push_animations") or not gui.block_push_animations:
        return

    for anim in gui.block_push_animations:
        progress = anim.get("progress", 0.0)
        eased = 1.0 - (1.0 - progress) ** 2

        from_r, from_c = anim["from_pos"]
        to_r, to_c = anim["to_pos"]
        cur_r = from_r + (to_r - from_r) * eased
        cur_c = from_c + (to_c - from_c) * eased

        screen_x = int(cur_c * gui.TILE_SIZE - gui.view_offset_x)
        screen_y = int(cur_r * gui.TILE_SIZE - gui.view_offset_y)

        block_id = semantic_palette["BLOCK"]
        if getattr(gui, "renderer", None) and hasattr(gui.renderer, "sprite_manager"):
            block_surf = gui.renderer.sprite_manager.get_tile(block_id, gui.TILE_SIZE)
        else:
            block_surf = gui.images.get(block_id)

        if block_surf:
            surface.blit(block_surf, (screen_x, screen_y))
            if progress < 1.0:
                highlight = pygame.Surface((gui.TILE_SIZE, gui.TILE_SIZE), pygame.SRCALPHA)
                alpha = int(60 * (1.0 - progress))
                highlight.fill((255, 255, 200, alpha))
                surface.blit(highlight, (screen_x, screen_y))


def get_animating_block_positions(gui: Any) -> set:
    """Return source positions for currently animating blocks."""
    if not hasattr(gui, "block_push_animations") or not gui.block_push_animations:
        return set()
    return {anim["from_pos"] for anim in gui.block_push_animations}


def check_and_start_block_push(gui: Any, player_pos: Tuple[int, int], target_pos: Tuple[int, int], walkable_ids: set, pushable_ids: set) -> bool:
    """Check whether move should push a block and start animation if valid."""
    if not hasattr(gui, "env") or gui.env is None:
        return False

    target_tile = gui.env.grid[target_pos[0], target_pos[1]]
    if target_tile not in pushable_ids:
        return False

    dr = target_pos[0] - player_pos[0]
    dc = target_pos[1] - player_pos[1]
    push_dest_r = target_pos[0] + dr
    push_dest_c = target_pos[1] + dc

    if not (0 <= push_dest_r < gui.env.height and 0 <= push_dest_c < gui.env.width):
        return False

    dest_tile = gui.env.grid[push_dest_r, push_dest_c]
    if dest_tile not in walkable_ids:
        return False

    gui._start_block_push_animation(target_pos, (push_dest_r, push_dest_c))
    gui._show_toast("Block pushed!", 1.5, "info")
    return True
