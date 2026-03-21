"""Render diagnostics and map-overlay helpers extracted from gui_runner."""

from __future__ import annotations

from typing import Any


def log_draw_ranges(
    *,
    gui: Any,
    start_r: int,
    end_r: int,
    start_c: int,
    end_c: int,
    h: int,
    w: int,
    time_module: Any,
    logger: Any,
) -> None:
    """Throttle and emit draw-range diagnostics to help blank-screen debugging."""
    if not hasattr(gui, "_last_render_log_time"):
        gui._last_render_log_time = 0.0

    current_time = time_module.time()
    if current_time - gui._last_render_log_time < 1.0:
        return

    gui._last_render_log_time = current_time
    try:
        logger.debug("Draw ranges r=%d..%d c=%d..%d (map h=%d w=%d)", start_r, end_r, start_c, end_c, h, w)
        if start_r < h and start_c < w:
            sample_tile = int(gui.env.grid[start_r, start_c])
            logger.debug(
                "Render sample tile: tile=%r, images_contains=%r",
                sample_tile,
                sample_tile in getattr(gui, "images", {}),
            )
    except Exception:
        logger.exception("Failed to log draw ranges")


def render_empty_range_warning(
    *,
    gui: Any,
    start_r: int,
    end_r: int,
    start_c: int,
    end_c: int,
    pygame: Any,
) -> None:
    """Render explicit on-screen warning when computed draw range is empty."""
    if (end_r > start_r) and (end_c > start_c):
        return

    try:
        pygame.draw.rect(gui.screen, (200, 0, 0), (gui.screen_w // 2 - 120, gui.screen_h // 2 - 40, 240, 80))
        font = gui.big_font if hasattr(gui, "big_font") else pygame.font.SysFont("Arial", 20, True)
        txt = font.render("RENDER RANGE EMPTY - CHECK OFFSETS", True, (255, 255, 255))
        gui.screen.blit(txt, (gui.screen_w // 2 - txt.get_width() // 2, gui.screen_h // 2 - txt.get_height() // 2))
    except Exception:
        pass


def render_jps_overlay(
    *,
    gui: Any,
    map_surface: Any,
    start_r: int,
    end_r: int,
    start_c: int,
    end_c: int,
    pygame: Any,
) -> None:
    """Render JPS path segments and jump points for debugging/teaching."""
    if not gui.feature_flags.get("show_jps_overlay", False):
        return
    if not getattr(gui, "last_jps_trace", None):
        return

    trace = gui.last_jps_trace
    jumps = trace.get("jumps", []) if trace else []
    segments = trace.get("segments", []) if trace else []

    for a, b in segments:
        ar, ac = a
        br, bc = b
        if not (start_r <= ar < end_r and start_c <= ac < end_c and start_r <= br < end_r and start_c <= bc < end_c):
            continue
        ax = ac * gui.TILE_SIZE + gui.TILE_SIZE // 2 - gui.view_offset_x
        ay = ar * gui.TILE_SIZE + gui.TILE_SIZE // 2 - gui.view_offset_y
        bx = bc * gui.TILE_SIZE + gui.TILE_SIZE // 2 - gui.view_offset_x
        by = br * gui.TILE_SIZE + gui.TILE_SIZE // 2 - gui.view_offset_y
        try:
            pygame.draw.line(map_surface, (255, 180, 0, 180), (ax, ay), (bx, by), 2)
        except Exception:
            pygame.draw.line(map_surface, (255, 180, 0), (ax, ay), (bx, by), 2)

    for jr, jc in jumps:
        if not (start_r <= jr < end_r and start_c <= jc < end_c):
            continue
        sx = jc * gui.TILE_SIZE - gui.view_offset_x
        sy = jr * gui.TILE_SIZE - gui.view_offset_y
        dot = pygame.Surface((8, 8), pygame.SRCALPHA)
        dot.fill((255, 100, 0, 200))
        map_surface.blit(dot, (sx + gui.TILE_SIZE // 2 - 4, sy + gui.TILE_SIZE // 2 - 4))


def render_map_elites_overlay(
    *,
    gui: Any,
    map_surface: Any,
    pygame: Any,
) -> None:
    """Render compact MAP-Elites occupancy overlay in map-space coordinates."""
    if not gui.feature_flags.get("show_map_elites", False):
        return
    if not getattr(gui, "map_elites_result", None):
        return

    try:
        occ = gui.map_elites_result.occupancy_grid()
        res = occ.shape[0]
        mini_w = 120
        mini_h = 120
        cell_w = max(2, mini_w // res)
        cell_h = max(2, mini_h // res)
        mini_surf = pygame.Surface((cell_w * res, cell_h * res))
        mini_surf.fill((30, 30, 40))
        for yy in range(res):
            for xx in range(res):
                if occ[yy, xx]:
                    pygame.draw.rect(mini_surf, (100, 200, 255), (xx * cell_w, (res - 1 - yy) * cell_h, cell_w, cell_h))

        sidebar_x = gui.screen_w - gui.SIDEBAR_WIDTH + 10
        blit_x = sidebar_x + 8
        blit_y = 120
        map_surface.blit(mini_surf, (blit_x - gui.view_offset_x, blit_y - gui.view_offset_y))
    except Exception:
        pass
