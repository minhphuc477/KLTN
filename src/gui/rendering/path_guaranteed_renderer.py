"""Guaranteed path overlay renderer extracted from gui_runner."""

from __future__ import annotations

from typing import Any


def render_path_guaranteed(
    *,
    gui: Any,
    surface: Any,
    pygame: Any,
    math_module: Any,
    time_module: Any,
    logger: Any,
) -> None:
    """Render path overlay regardless of mode/feature flag state."""
    path = getattr(gui, "auto_path", None)
    test_path = getattr(gui, "_test_path", None)

    if not path or len(path) < 1:
        path = test_path
        if not path or len(path) < 1:
            return
        is_test = True
    else:
        is_test = False

    try:
        for i, point in enumerate(path):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                logger.warning("Invalid path point at index %d: %s", i, point)
                return
            if not all(isinstance(coord, (int, float)) for coord in point):
                logger.warning("Invalid path coordinates at index %d: %s", i, point)
                return
    except Exception as exc:
        logger.warning("Path validation failed: %s", exc)
        return

    path_len = len(path)
    tile_size = gui.TILE_SIZE
    vx = gui.view_offset_x
    vy = gui.view_offset_y

    if is_test:
        line_color = (255, 0, 0)
        outline_color = (128, 0, 0)
        start_color = (255, 100, 100)
        end_color = (255, 50, 50)
    else:
        line_color = (0, 255, 255)
        outline_color = (0, 0, 0)
        start_color = (0, 255, 0)
        end_color = (255, 215, 0)

    if path_len > 1:
        for i in range(path_len - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            sx1 = int(c1 * tile_size - vx + tile_size // 2)
            sy1 = int(r1 * tile_size - vy + tile_size // 2)
            sx2 = int(c2 * tile_size - vx + tile_size // 2)
            sy2 = int(r2 * tile_size - vy + tile_size // 2)

            pygame.draw.line(surface, outline_color, (sx1, sy1), (sx2, sy2), 7)
            pygame.draw.line(surface, line_color, (sx1, sy1), (sx2, sy2), 5)

    sr, sc = path[0]
    cx = int(sc * tile_size - vx + tile_size // 2)
    cy = int(sr * tile_size - vy + tile_size // 2)
    pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 10)
    pygame.draw.circle(surface, start_color, (cx, cy), 8)

    er, ec = path[-1]
    ecx = int(ec * tile_size - vx + tile_size // 2)
    ecy = int(er * tile_size - vy + tile_size // 2)
    pygame.draw.circle(surface, (0, 0, 0), (ecx, ecy), 10)
    pygame.draw.circle(surface, end_color, (ecx, ecy), 8)

    path_item_positions = getattr(gui, "path_item_positions", {})
    collected_positions = getattr(gui, "collected_positions", set())
    current_step = getattr(gui, "auto_step_idx", 0)

    item_colors = {
        "keys": (255, 215, 0),
        "boss_keys": (255, 100, 50),
        "ladders": (100, 200, 255),
        "bombs": (150, 150, 150),
        "doors_locked": (139, 69, 19),
        "doors_bomb": (80, 80, 80),
        "doors_boss": (180, 40, 40),
        "triforce": (255, 255, 100),
    }

    pulse = (math_module.sin(time_module.time() * 4) + 1) / 2

    for item_type, positions in path_item_positions.items():
        if not positions:
            continue

        color = item_colors.get(item_type, (255, 255, 255))

        for pos in positions:
            if pos in collected_positions:
                continue

            try:
                item_path_idx = path.index(pos) if pos in path else -1
            except ValueError:
                item_path_idx = -1

            if item_path_idx >= 0 and item_path_idx <= current_step:
                continue

            r, c = pos
            ix = int(c * tile_size - vx + tile_size // 2)
            iy = int(r * tile_size - vy + tile_size // 2)

            ring_size = int(12 + 4 * pulse)
            pygame.draw.circle(surface, (0, 0, 0), (ix, iy), ring_size + 2, 3)
            pygame.draw.circle(surface, color, (ix, iy), ring_size, 3)
            pygame.draw.circle(surface, (255, 255, 255), (ix, iy), 4)

    if not hasattr(gui, "_guaranteed_path_log_time"):
        gui._guaranteed_path_log_time = 0
    now = time_module.time()
    if now - gui._guaranteed_path_log_time > 2.0:
        gui._guaranteed_path_log_time = now
        items_count = sum(len(v) for v in path_item_positions.values()) if path_item_positions else 0
        logger.debug(
            "GUARANTEED_PATH: Rendered %d segments, %d item markers, start=(%d,%d)->screen(%d,%d), end=(%d,%d)->screen(%d,%d), is_test=%s",
            path_len - 1,
            items_count,
            path[0][0],
            path[0][1],
            cx,
            cy,
            path[-1][0],
            path[-1][1],
            ecx,
            ecy,
            is_test,
        )
