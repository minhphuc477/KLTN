"""Sidebar rendering section helpers extracted from gui_runner._render."""

from __future__ import annotations

from typing import Any


def render_sidebar_header_inventory_solver(
    *,
    gui: Any,
    screen: Any,
    sidebar_x: int,
    y_pos: int,
    map_w: int,
    map_h: int,
    time_module: Any,
    math_module: Any,
    pygame: Any,
    logger: Any,
) -> int:
    """Render sidebar title/map/inventory and optional solver analysis; return updated y."""
    title = gui.big_font.render("ZAVE", True, (100, 200, 255))
    screen.blit(title, (sidebar_x + 10, y_pos))
    y_pos += 28

    if gui.current_map_idx < len(gui.map_names):
        name = gui.map_names[gui.current_map_idx]
    else:
        name = f"Map {gui.current_map_idx + 1}"
    name_surf = gui.font.render(name, True, (255, 220, 100))
    screen.blit(name_surf, (sidebar_x + 10, y_pos))
    y_pos += 20

    map_num = f"({gui.current_map_idx + 1}/{len(gui.maps)})"
    num_surf = gui.small_font.render(map_num, True, (150, 150, 150))
    screen.blit(num_surf, (sidebar_x + 10, y_pos))
    y_pos += 18

    size_info = f"Size: {map_w}x{map_h}"
    size_surf = gui.small_font.render(size_info, True, (150, 150, 150))
    screen.blit(size_surf, (sidebar_x + 10, y_pos))
    y_pos += 20

    pygame.draw.line(screen, (60, 60, 80), (sidebar_x + 10, y_pos), (gui.screen_w - 10, y_pos))
    y_pos += 10

    inv_title = gui.font.render("Inventory", True, (255, 200, 100))
    screen.blit(inv_title, (sidebar_x + 10, y_pos))
    y_pos += 22

    current_time = time_module.time()
    key_highlight = "key" in gui.item_pickup_times and (current_time - gui.item_pickup_times["key"]) < 1.0
    bomb_highlight = "bomb" in gui.item_pickup_times and (current_time - gui.item_pickup_times["bomb"]) < 1.0
    boss_key_highlight = "boss_key" in gui.item_pickup_times and (current_time - gui.item_pickup_times["boss_key"]) < 1.0

    gui._sync_inventory_counters()

    if hasattr(gui, "_inv_render_frame_count"):
        gui._inv_render_frame_count = getattr(gui, "_inv_render_frame_count", 0) + 1
    else:
        gui._inv_render_frame_count = 0
    if gui._inv_render_frame_count % 60 == 0:
        logger.debug(
            "INVENTORY_RENDER: keys_collected=%d, total_keys=%d, env.state.keys=%d, bombs_collected=%d, has_bomb=%s, boss_keys_collected=%d, has_boss_key=%s, collected_items_len=%d",
            gui.keys_collected,
            gui.total_keys,
            gui.env.state.keys,
            gui.bombs_collected,
            gui.env.state.has_bomb,
            gui.boss_keys_collected,
            getattr(gui.env.state, "has_boss_key", False),
            len(gui.collected_items),
        )

    if gui.total_keys > 0:
        keys_text = f"Keys: {gui.keys_collected}/{gui.total_keys} ({gui.env.state.keys} held)"
    else:
        keys_text = f"Keys: {gui.env.state.keys}"

    if getattr(gui, "last_pickup_msg", None):
        hint = gui.small_font.render(gui.last_pickup_msg, True, (200, 200, 200))
        screen.blit(hint, (sidebar_x + 15, y_pos))
        y_pos += 16
    if getattr(gui, "last_use_msg", None):
        hint2 = gui.small_font.render(gui.last_use_msg, True, (200, 200, 200))
        screen.blit(hint2, (sidebar_x + 15, y_pos))
        y_pos += 16

    if key_highlight:
        flash_alpha = (math_module.sin(current_time * 15) + 1) / 2
        keys_color = (255, int(220 + 35 * flash_alpha), int(100 + 155 * flash_alpha))
    else:
        keys_color = (255, 220, 100)
    keys_surf = gui.small_font.render(keys_text, True, keys_color)
    screen.blit(keys_surf, (sidebar_x + 15, y_pos))
    y_pos += 18

    if gui.total_bombs > 0:
        bomb_text = f"Bombs: {gui.bombs_collected}/{gui.total_bombs} {'[Y]' if gui.env.state.has_bomb else '[N]'}"
    else:
        bomb_text = f"Bomb: {'[Y]' if gui.env.state.has_bomb else '[N]'}"
    if bomb_highlight:
        flash_alpha = (math_module.sin(current_time * 15) + 1) / 2
        bomb_color = (int(200 + 55 * flash_alpha), int(80 + 175 * flash_alpha), int(80 + 175 * flash_alpha))
    else:
        bomb_color = (100, 255, 100) if gui.env.state.has_bomb else (150, 150, 150)
    bomb_surf = gui.small_font.render(bomb_text, True, bomb_color)
    screen.blit(bomb_surf, (sidebar_x + 15, y_pos))
    y_pos += 18

    if gui.total_boss_keys > 0:
        boss_key_text = f"Boss Key: {gui.boss_keys_collected}/{gui.total_boss_keys} {'[Y]' if gui.env.state.has_boss_key else '[N]'}"
    else:
        boss_key_text = f"Boss Key: {'[Y]' if gui.env.state.has_boss_key else '[N]'}"
    if boss_key_highlight:
        flash_alpha = (math_module.sin(current_time * 15) + 1) / 2
        boss_color = (int(180 + 75 * flash_alpha), int(40 + 215 * flash_alpha), int(180 + 75 * flash_alpha))
    else:
        boss_color = (255, 150, 100) if gui.env.state.has_boss_key else (150, 150, 150)
    boss_surf = gui.small_font.render(boss_key_text, True, boss_color)
    screen.blit(boss_surf, (sidebar_x + 15, y_pos))
    y_pos += 18

    if gui.solver_result:
        y_pos += 5
        pygame.draw.line(screen, (60, 60, 80), (sidebar_x + 10, y_pos), (gui.screen_w - 10, y_pos))
        y_pos += 8

        solver_title = gui.font.render("Path Analysis", True, (100, 200, 255))
        screen.blit(solver_title, (sidebar_x + 10, y_pos))
        y_pos += 20

        keys_avail = gui.solver_result.get("keys_available", 0)
        keys_used = gui.solver_result.get("keys_used", 0)
        key_info = f"Keys: {keys_avail} found, {keys_used} used"
        key_color = (255, 220, 100) if keys_used > 0 else (150, 200, 150)
        key_surf = gui.small_font.render(key_info, True, key_color)
        screen.blit(key_surf, (sidebar_x + 15, y_pos))
        y_pos += 16

        edge_types_raw = gui.solver_result.get("edge_types")
        if isinstance(edge_types_raw, (list, tuple, set)):
            edge_types = list(edge_types_raw)
        else:
            edge_types = []
        if edge_types:
            type_counts = {}
            for et in edge_types:
                type_counts[et] = type_counts.get(et, 0) + 1

            edge_colors = {
                "open": (100, 255, 100),
                "key_locked": (255, 220, 100),
                "bombable": (255, 150, 50),
                "soft_locked": (180, 100, 255),
                "stair": (100, 200, 255),
            }

            for etype, count in type_counts.items():
                color = edge_colors.get(etype, (150, 150, 150))
                type_name = etype.replace("_", " ").title()
                et_text = f"  {type_name}: {count}"
                et_surf = gui.small_font.render(et_text, True, color)
                screen.blit(et_surf, (sidebar_x + 15, y_pos))
                y_pos += 14

    y_pos += 7
    return y_pos


def render_sidebar_status_message_metrics_controls(
    *,
    gui: Any,
    screen: Any,
    sidebar_x: int,
    y_pos: int,
    player_row: int,
    player_col: int,
    pygame: Any,
    time_module: Any,
    math_module: Any,
    semantic_palette: dict,
) -> int:
    """Render status/message/metrics/controls sections; return updated y."""
    pygame.draw.line(screen, (60, 60, 80), (sidebar_x + 10, y_pos), (gui.screen_w - 10, y_pos))
    y_pos += 10

    status_title = gui.font.render("STATUS", True, (180, 220, 255))
    screen.blit(status_title, (sidebar_x + 10, y_pos))
    y_pos += 22

    if gui.env:
        map_name = gui.map_names[gui.current_map_idx] if gui.current_map_idx < len(gui.map_names) else f"Map {gui.current_map_idx + 1}"
        pos = gui.env.state.position
        status_text = f"{map_name}"
        status_surf = gui.small_font.render(status_text, True, (200, 220, 255))
        screen.blit(status_surf, (sidebar_x + 15, y_pos))
        y_pos += 16

        pos_text = f"Pos: ({pos[0]}, {pos[1]})"
        pos_surf = gui.small_font.render(pos_text, True, (150, 150, 150))
        screen.blit(pos_surf, (sidebar_x + 15, y_pos))
        y_pos += 16

        ready_text = "Auto-solving" if gui.auto_mode else "Ready"
        ready_color = (100, 255, 100) if gui.auto_mode else (200, 200, 200)
        ready_surf = gui.small_font.render(ready_text, True, ready_color)
        screen.blit(ready_surf, (sidebar_x + 15, y_pos))
        y_pos += 18

        try:
            if getattr(gui, "dstar_active", False) and getattr(gui, "dstar_solver", None):
                replans = getattr(gui.dstar_solver, "replans_count", 0)
                ds_text = f"D* Lite: ACTIVE ({replans} replans)"
                ds_surf = gui.small_font.render(ds_text, True, (100, 220, 255))
                screen.blit(ds_surf, (sidebar_x + 15, y_pos))
                y_pos += 16
            elif gui.feature_flags.get("dstar_lite", False):
                ds_text = "D* Lite: enabled"
                ds_surf = gui.small_font.render(ds_text, True, (180, 180, 255))
                screen.blit(ds_surf, (sidebar_x + 15, y_pos))
                y_pos += 16
        except Exception:
            pass

        try:
            if hasattr(gui.env, "_find_all_positions"):
                stair_positions = list(map(tuple, gui.env._find_all_positions(semantic_palette["STAIR"])))
            else:
                stair_positions = []
            stair_count = len(stair_positions)
            stair_text = f"Stairs: {stair_count} | Sprite: {'Full' if getattr(gui, 'stair_sprite', None) else 'No'}"
            stair_surf = gui.small_font.render(stair_text, True, (200, 200, 150))
            screen.blit(stair_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
        except Exception:
            pass

    pygame.draw.line(screen, (60, 60, 80), (sidebar_x + 10, y_pos), (gui.screen_w - 10, y_pos))
    y_pos += 10

    message_title = gui.font.render("MESSAGE", True, (100, 255, 150))
    screen.blit(message_title, (sidebar_x + 10, y_pos))
    y_pos += 22

    if gui.message and (time_module.time() - gui.message_time) < gui.message_duration:
        elapsed = time_module.time() - gui.message_time
        remaining = gui.message_duration - elapsed
        alpha = min(1.0, remaining / 0.5) if remaining < 0.5 else 1.0

        max_chars = 28
        words = gui.message.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for line in lines[:3]:
            msg_color = tuple(int(c * alpha) for c in (150, 255, 200))
            msg_surf = gui.small_font.render(line, True, msg_color)
            screen.blit(msg_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
    else:
        default_msg = "Press SPACE to solve"
        msg_surf = gui.small_font.render(default_msg, True, (120, 120, 120))
        screen.blit(msg_surf, (sidebar_x + 15, y_pos))
        y_pos += 16

    y_pos += 8

    pygame.draw.line(screen, (60, 60, 80), (sidebar_x + 10, y_pos), (gui.screen_w - 10, y_pos))
    y_pos += 10

    metrics_title = gui.font.render("Metrics", True, (150, 200, 255))
    screen.blit(metrics_title, (sidebar_x + 10, y_pos))
    y_pos += 22

    steps_text = f"Steps: {gui.step_count}"
    steps_surf = gui.small_font.render(steps_text, True, (200, 200, 200))
    screen.blit(steps_surf, (sidebar_x + 15, y_pos))
    y_pos += 16

    speed_color = (100, 255, 100) if gui.speed_multiplier == 1.0 else (255, 200, 100)
    speed_text = f"Speed: {gui.speed_multiplier}x"
    speed_surf = gui.small_font.render(speed_text, True, speed_color)
    screen.blit(speed_surf, (sidebar_x + 15, y_pos))
    y_pos += 16

    zoom_text = f"Zoom: {gui.TILE_SIZE}px"
    zoom_surf = gui.small_font.render(zoom_text, True, (150, 150, 150))
    screen.blit(zoom_surf, (sidebar_x + 15, y_pos))
    y_pos += 16

    fps = int(gui.clock.get_fps())
    fps_color = (100, 255, 100) if fps >= 25 else (255, 150, 150)
    fps_text = f"FPS: {fps}"
    fps_surf = gui.small_font.render(fps_text, True, fps_color)
    screen.blit(fps_surf, (sidebar_x + 15, y_pos))
    y_pos += 18

    pygame.draw.line(screen, (60, 60, 80), (sidebar_x + 10, y_pos), (gui.screen_w - 10, y_pos))
    y_pos += 10

    ctrl_title = gui.font.render("Controls", True, (100, 200, 100))
    screen.blit(ctrl_title, (sidebar_x + 10, y_pos))
    y_pos += 20

    controls = [
        "Arrows Move",
        "SPACE Auto-solve",
        "R     Reset map",
        "N/P   Next/Prev",
        "+/-   Zoom",
        "0     Reset zoom",
        "C     Center view",
        "M     Minimap",
        "[/]   Speed+/-",
        "F11   Fullscreen",
        "F1    Help",
        "H     Heatmap",
        "ESC   Quit",
    ]

    for ctrl in controls:
        ctrl_surf = gui.small_font.render(ctrl, True, (120, 120, 120))
        screen.blit(ctrl_surf, (sidebar_x + 15, y_pos))
        y_pos += 15

    hud_y = gui.screen_h - gui.HUD_HEIGHT
    pygame.draw.rect(screen, (30, 30, 45), (0, hud_y, gui.screen_w - gui.SIDEBAR_WIDTH, gui.HUD_HEIGHT))
    pygame.draw.line(screen, (60, 60, 80), (0, hud_y), (gui.screen_w - gui.SIDEBAR_WIDTH, hud_y), 2)

    msg_color = (255, 255, 100) if gui.env.won else (200, 200, 200)
    msg_surf = gui.font.render(gui.message, True, msg_color)
    screen.blit(msg_surf, (10, hud_y + 10))

    pos_text = f"Position: ({player_row}, {player_col})"
    pos_surf = gui.small_font.render(pos_text, True, (150, 150, 150))
    screen.blit(pos_surf, (10, hud_y + 35))

    if gui.env.won:
        win_text = "*** VICTORY! ***"
        win_surf = gui.big_font.render(win_text, True, (255, 215, 0))
        screen.blit(win_surf, (10, hud_y + 55))

    return y_pos
