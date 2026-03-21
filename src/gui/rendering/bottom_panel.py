"""Bottom panel rendering helpers extracted from gui_runner."""

from typing import Any


def render_unified_bottom_panel(gui: Any, pygame: Any) -> None:
    """Render unified bottom HUD panel with status and message sections."""
    panel_height = 80
    panel_y = gui.screen_h - panel_height - 5
    panel_x = 5
    panel_width = gui.screen_w - gui.SIDEBAR_WIDTH - 15

    panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_bg = pygame.Rect(0, 0, panel_width, panel_height)
    pygame.draw.rect(panel_surf, (35, 35, 50, 230), panel_bg, border_radius=8)
    pygame.draw.rect(panel_surf, (60, 60, 80), panel_bg, 2, border_radius=8)
    gui.screen.blit(panel_surf, (panel_x, panel_y))

    padding = 15
    total_inner_width = panel_width - (padding * 2)
    status_width = int(total_inner_width * 0.50)
    message_width = total_inner_width - status_width

    status_x = panel_x + padding
    message_x = status_x + status_width + 20
    content_y = panel_y + 10
    section_height = panel_height - 20

    divider_color = (60, 60, 80)
    pygame.draw.line(
        gui.screen,
        divider_color,
        (message_x - 10, content_y),
        (message_x - 10, content_y + section_height),
        2,
    )

    render_status_section(gui, status_x, content_y, status_width, section_height)
    render_message_section(gui, message_x, content_y, message_width, section_height)


def render_message_section(gui: Any, x: int, y: int, width: int, height: int) -> None:
    """Render message/status section in bottom panel."""
    _ = (width, height)
    title_surf = gui.font.render("MESSAGE", True, (100, 255, 150))
    gui.screen.blit(title_surf, (x, y))
    y += 22

    msg_color = (255, 255, 100) if (gui.env and gui.env.won) else (200, 200, 200)
    if len(gui.message) > 35:
        msg_lines = [gui.message[i:i + 35] for i in range(0, len(gui.message), 35)]
        for line in msg_lines[:2]:
            msg_surf = gui.small_font.render(line, True, msg_color)
            gui.screen.blit(msg_surf, (x, y))
            y += 16
    else:
        msg_surf = gui.small_font.render(gui.message, True, msg_color)
        gui.screen.blit(msg_surf, (x, y))


def render_progress_bar(
    surface: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    filled: int,
    total: int,
    color_filled: tuple,
    color_empty: tuple,
    pygame: Any,
) -> None:
    """Render segmented progress bar with filled and empty segments."""
    if total == 0:
        return

    segments = min(total, 10)
    segment_width = width // max(1, segments)
    items_per_segment = total / segments

    for i in range(segments):
        segment_x = x + i * segment_width
        segment_rect = pygame.Rect(segment_x + 1, y, segment_width - 2, height)
        segment_threshold = (i + 1) * items_per_segment
        is_filled = filled >= segment_threshold

        if is_filled:
            pygame.draw.rect(surface, color_filled, segment_rect, border_radius=2)
            highlight = tuple(min(c + 40, 255) for c in color_filled[:3])
            pygame.draw.rect(surface, highlight, segment_rect, 1, border_radius=2)
        else:
            pygame.draw.rect(surface, color_empty, segment_rect, border_radius=2)
            pygame.draw.rect(surface, (60, 60, 80), segment_rect, 1, border_radius=2)


def render_inventory_section(gui: Any, x: int, y: int, width: int, height: int, pygame: Any, time_module: Any, logger: Any) -> None:
    """Render inventory section with progress bars and item status."""
    _ = height
    try:
        gui._sync_inventory_counters()
    except Exception:
        pass

    logger.debug(
        "INVENTORY_RENDER: keys_collected=%d, total_keys=%d, env.state.keys=%d, "
        "bombs_collected=%d, has_bomb=%s, boss_keys_collected=%d, has_boss_key=%s, "
        "collected_items_len=%d",
        gui.keys_collected,
        gui.total_keys,
        getattr(gui.env.state, "keys", 0) if gui.env and gui.env.state else 0,
        gui.bombs_collected,
        getattr(gui.env.state, "has_bomb", False) if gui.env and gui.env.state else False,
        gui.boss_keys_collected,
        getattr(gui.env.state, "has_boss_key", False) if gui.env and gui.env.state else False,
        len(gui.collected_items) if hasattr(gui, "collected_items") else 0,
    )

    title_surf = gui.font.render("INVENTORY", True, (100, 200, 255))
    gui.screen.blit(title_surf, (x, y))

    y_offset = y + 25
    line_height = 24
    bar_width = width - 10
    bar_height = 8

    if not gui.env:
        return

    held_keys = gui.env.state.keys if hasattr(gui.env.state, "keys") else 0
    keys_color = (255, 215, 0)

    current_time = time_module.time()
    for _, pickup_time in list(gui.item_pickup_times.items()):
        if current_time - pickup_time < 0.5:
            keys_color = (255, 255, 150)
            break

    keys_text = f"K: {gui.keys_collected}/{gui.total_keys}"
    if held_keys > 0:
        keys_text += f" ({held_keys} held)"
    keys_surf = gui.small_font.render(keys_text, True, keys_color)
    gui.screen.blit(keys_surf, (x, y_offset))

    if gui.total_keys > 0:
        render_progress_bar(
            gui.screen,
            x,
            y_offset + 16,
            bar_width,
            bar_height,
            gui.keys_collected,
            gui.total_keys,
            keys_color,
            (40, 40, 50),
            pygame,
        )
    y_offset += line_height + 10

    has_bomb = hasattr(gui.env.state, "has_bomb") and gui.env.state.has_bomb
    bombs_color = (255, 107, 53) if has_bomb else (100, 100, 100)
    bombs_status = "[YES]" if has_bomb else "[NO]"

    if gui.total_bombs > 0:
        bombs_text = f"B: {bombs_status} Bomb"
        bombs_surf = gui.small_font.render(bombs_text, True, bombs_color)
        gui.screen.blit(bombs_surf, (x, y_offset))
        y_offset += line_height

    has_boss_key = hasattr(gui.env.state, "has_boss_key") and gui.env.state.has_boss_key
    boss_key_color = (176, 66, 255) if has_boss_key else (100, 100, 100)

    if gui.total_boss_keys > 0:
        boss_key_text = f"Boss Key: {gui.boss_keys_collected}/{gui.total_boss_keys}"
        if has_boss_key:
            boss_key_text += " [Y]"
        boss_key_surf = gui.small_font.render(boss_key_text, True, boss_key_color)
        gui.screen.blit(boss_key_surf, (x, y_offset))

        render_progress_bar(
            gui.screen,
            x,
            y_offset + 16,
            bar_width,
            bar_height,
            gui.boss_keys_collected,
            gui.total_boss_keys,
            boss_key_color,
            (40, 40, 50),
            pygame,
        )


def render_metrics_section(gui: Any, x: int, y: int, width: int, height: int) -> None:
    """Render metrics section with steps, speed, zoom and environment steps."""
    _ = (width, height)
    title_surf = gui.font.render("METRICS", True, (150, 200, 255))
    gui.screen.blit(title_surf, (x, y))

    y_offset = y + 25
    line_height = 20

    steps_surf = gui.small_font.render(f"Steps: {gui.step_count}", True, (200, 200, 200))
    gui.screen.blit(steps_surf, (x, y_offset))
    y_offset += line_height

    speed_color = (100, 255, 100) if gui.speed_multiplier == 1.0 else (255, 200, 100)
    speed_surf = gui.small_font.render(f"Speed: {gui.speed_multiplier}x", True, speed_color)
    gui.screen.blit(speed_surf, (x, y_offset))
    y_offset += line_height

    zoom_surf = gui.small_font.render(f"Zoom: {gui.TILE_SIZE}px", True, (150, 150, 150))
    gui.screen.blit(zoom_surf, (x, y_offset))
    y_offset += line_height

    env_steps = gui.env.step_count if gui.env and hasattr(gui.env, "step_count") else 0
    env_surf = gui.small_font.render(f"Env: {env_steps}", True, (150, 150, 150))
    gui.screen.blit(env_surf, (x, y_offset))


def render_controls_section(gui: Any, x: int, y: int, width: int, height: int) -> None:
    """Render controls section using a two-column compact layout."""
    _ = height
    title_surf = gui.font.render("CONTROLS", True, (100, 200, 100))
    gui.screen.blit(title_surf, (x, y))

    y_offset = y + 25
    line_height = 16
    col_width = width // 2

    controls_left = [
        ("ARROWS", "Move"),
        ("SPACE", "Solve"),
        ("R", "Reset"),
        ("N/P", "Maps"),
        ("[/]", "Speed"),
    ]
    controls_right = [
        ("M", "Minimap"),
        ("H", "Heatmap"),
        ("+/-", "Zoom"),
        ("F11", "Full"),
        ("ESC", "Quit"),
    ]

    text_color = (120, 120, 120)
    for key, desc in controls_left:
        control_surf = gui.small_font.render(f"{key:4s} {desc}", True, text_color)
        gui.screen.blit(control_surf, (x, y_offset))
        y_offset += line_height

    y_offset = y + 25
    for key, desc in controls_right:
        control_surf = gui.small_font.render(f"{key:4s} {desc}", True, text_color)
        gui.screen.blit(control_surf, (x + col_width, y_offset))
        y_offset += line_height


def render_status_section(gui: Any, x: int, y: int, width: int, height: int) -> None:
    """Render status section with map, position and run state."""
    _ = (width, height)
    title_surf = gui.font.render("STATUS", True, (180, 220, 255))
    gui.screen.blit(title_surf, (x, y))

    y_offset = y + 25
    line_height = 18

    if gui.env and gui.env.won:
        status_text = "*** VICTORY! ***"
        status_color = (255, 215, 0)
        status_surf = gui.big_font.render(status_text, True, status_color)
        gui.screen.blit(status_surf, (x, y_offset))
        return

    if gui.current_map_idx < len(gui.map_names):
        map_name = gui.map_names[gui.current_map_idx]
        map_text = f"Map: {map_name[:15]}"
        map_surf = gui.small_font.render(map_text, True, (150, 200, 255))
        gui.screen.blit(map_surf, (x, y_offset))
        y_offset += line_height

    if gui.env and hasattr(gui.env.state, "position"):
        pos = gui.env.state.position
        pos_text = f"Pos: ({pos[0]}, {pos[1]})"
        pos_surf = gui.small_font.render(pos_text, True, (150, 150, 150))
        gui.screen.blit(pos_surf, (x, y_offset))
        y_offset += line_height

    if gui.auto_mode and gui.auto_path:
        progress_text = f"Auto: {gui.auto_step_idx}/{len(gui.auto_path)}"
        progress_surf = gui.small_font.render(progress_text, True, (100, 255, 150))
        gui.screen.blit(progress_surf, (x, y_offset))
        y_offset += line_height

    if gui.status_message:
        status_surf = gui.small_font.render(gui.status_message[:20], True, (180, 220, 255))
        gui.screen.blit(status_surf, (x, y_offset))
