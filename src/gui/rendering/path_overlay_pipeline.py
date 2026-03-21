"""Path overlay helpers extracted from gui_runner._render."""

from __future__ import annotations

from typing import Any


def render_planned_path_overlay(
    *,
    gui: Any,
    map_surface: Any,
    pygame: Any,
    math_module: Any,
    time_module: Any,
    logger: Any,
) -> None:
    """Render the explicit planned/visited path overlay used during auto-solve."""
    show_path = gui.auto_path and len(gui.auto_path) > 0

    if not hasattr(gui, "_path_render_log_counter"):
        gui._path_render_log_counter = 0
    gui._path_render_log_counter += 1

    if gui._path_render_log_counter % 120 == 1:
        logger.debug(
            "DEBUG_RENDER: show_path=%s, auto_path=%s, len=%d, auto_mode=%s, preview_visible=%s",
            show_path,
            bool(gui.auto_path),
            len(gui.auto_path) if gui.auto_path else 0,
            gui.auto_mode,
            getattr(gui, "preview_overlay_visible", False),
        )
        if gui.auto_path and len(gui.auto_path) > 0:
            logger.debug(
                "DEBUG_RENDER: Path first=%s, last=%s, step_idx=%d, view_offset=(%d,%d)",
                gui.auto_path[0],
                gui.auto_path[-1],
                getattr(gui, "auto_step_idx", 0),
                getattr(gui, "view_offset_x", 0),
                getattr(gui, "view_offset_y", 0),
            )

    if not show_path:
        return

    logger.debug(
        "Drawing path overlay: %d points, auto_mode=%s, step_idx=%s",
        len(gui.auto_path),
        gui.auto_mode,
        gui.auto_step_idx,
    )

    if len(gui.auto_path) > 1:
        for i in range(len(gui.auto_path) - 1):
            r1, c1 = gui.auto_path[i]
            r2, c2 = gui.auto_path[i + 1]
            x1 = int(c1 * gui.TILE_SIZE - gui.view_offset_x + gui.TILE_SIZE // 2)
            y1 = int(r1 * gui.TILE_SIZE - gui.view_offset_y + gui.TILE_SIZE // 2)
            x2 = int(c2 * gui.TILE_SIZE - gui.view_offset_x + gui.TILE_SIZE // 2)
            y2 = int(r2 * gui.TILE_SIZE - gui.view_offset_y + gui.TILE_SIZE // 2)

            if i >= gui.auto_step_idx:
                pygame.draw.line(map_surface, (0, 0, 0), (x1, y1), (x2, y2), 5)
                pygame.draw.line(map_surface, (0, 255, 255), (x1, y1), (x2, y2), 3)
            else:
                pygame.draw.line(map_surface, (0, 0, 0), (x1, y1), (x2, y2), 6)
                pygame.draw.line(map_surface, (0, 255, 0), (x1, y1), (x2, y2), 4)

    start_r, start_c = gui.auto_path[0]
    start_x = int(start_c * gui.TILE_SIZE - gui.view_offset_x + gui.TILE_SIZE // 2)
    start_y = int(start_r * gui.TILE_SIZE - gui.view_offset_y + gui.TILE_SIZE // 2)
    pygame.draw.circle(map_surface, (0, 0, 0), (start_x, start_y), 10)
    pygame.draw.circle(map_surface, (0, 255, 100), (start_x, start_y), 8)

    end_r, end_c = gui.auto_path[-1]
    end_x = int(end_c * gui.TILE_SIZE - gui.view_offset_x + gui.TILE_SIZE // 2)
    end_y = int(end_r * gui.TILE_SIZE - gui.view_offset_y + gui.TILE_SIZE // 2)
    pygame.draw.circle(map_surface, (0, 0, 0), (end_x, end_y), 10)
    pygame.draw.circle(map_surface, (255, 215, 0), (end_x, end_y), 8)

    if gui.auto_mode and gui.auto_step_idx > 0:
        for i, pos in enumerate(gui.auto_path[: gui.auto_step_idx + 1]):
            pr, pc = pos
            path_surf = pygame.Surface((gui.TILE_SIZE, gui.TILE_SIZE), pygame.SRCALPHA)

            current_time = time_module.time()
            is_recent_unlock = pos in gui.door_unlock_times and (current_time - gui.door_unlock_times[pos]) < 2.0

            if is_recent_unlock:
                flash_alpha = (math_module.sin(current_time * 8) + 1) / 2
                alpha = int(150 + 105 * flash_alpha)
                path_surf.fill((255, 215, 0, alpha))
            else:
                alpha = 40 + int(20 * (i / max(1, len(gui.auto_path))))
                path_surf.fill((0, 255, 0, alpha))

            screen_x = pc * gui.TILE_SIZE - gui.view_offset_x
            screen_y = pr * gui.TILE_SIZE - gui.view_offset_y
            map_surface.blit(path_surf, (screen_x, screen_y))
