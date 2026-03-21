"""Inventory display helpers for legend and path item summary text."""

from typing import Any


def get_path_items_display_text(gui: Any) -> str:
    """Build compact display text for path items summary."""
    summary = getattr(gui, "path_items_summary", {})
    if not summary:
        return ""

    parts = []
    if summary.get("keys", 0) > 0:
        parts.append(f"{summary['keys']}[K]")
    if summary.get("boss_keys", 0) > 0:
        parts.append(f"{summary['boss_keys']}[BK]")
    if summary.get("ladders", 0) > 0:
        parts.append(f"{summary['ladders']}[L]")
    if summary.get("bombs", 0) > 0:
        parts.append(f"{summary['bombs']}[B]")
    if summary.get("doors_locked", 0) > 0:
        parts.append(f"{summary['doors_locked']}[D]")
    if summary.get("doors_bomb", 0) > 0:
        parts.append(f"{summary['doors_bomb']}[Bx]")
    if summary.get("doors_boss", 0) > 0:
        parts.append(f"{summary['doors_boss']}[BD]")
    if summary.get("triforce", 0) > 0:
        parts.append(f"{summary['triforce']}[T]")

    return " ".join(parts) if parts else ""


def render_item_legend(gui: Any, surface: Any, pygame: Any) -> None:
    """Render inventory legend with current counters and path-ahead preview."""
    if not gui.env:
        return
    gui._sync_inventory_counters()

    path_summary = getattr(gui, "path_items_summary", {})
    has_path_items = any(v > 0 for v in path_summary.values()) if path_summary else False

    base_lines = 3
    path_item_lines = 1 if has_path_items else 0
    total_lines = base_lines + path_item_lines

    legend_x = 10
    legend_height = 20 + (total_lines * 20)
    legend_y = gui.screen_h - legend_height - 40

    legend_bg = pygame.Surface((350, legend_height), pygame.SRCALPHA)
    legend_bg.fill((30, 30, 40, 220))
    surface.blit(legend_bg, (legend_x, legend_y))
    pygame.draw.rect(surface, (70, 70, 100), (legend_x, legend_y, 350, legend_height), 2)

    title_surf = gui.small_font.render("Inventory", True, (100, 200, 255))
    surface.blit(title_surf, (legend_x + 10, legend_y + 4))

    y_offset = legend_y + 24
    legend_text = [
        f"[K] Keys: {gui.env.state.keys} held | {gui.keys_collected}/{gui.total_keys} collected | {getattr(gui, 'keys_used', 0)} used",
        f"[B] Bombs: {getattr(gui.env.state, 'bomb_count', 0)} held | {gui.bombs_collected}/{gui.total_bombs} collected | {getattr(gui, 'bombs_used', 0)} used",
        f"[BK] Boss Key: {'Yes' if getattr(gui.env.state, 'has_boss_key', False) else 'No'} | {gui.boss_keys_collected}/{gui.total_boss_keys} collected",
    ]
    for text in legend_text:
        text_surf = gui.small_font.render(text, True, (255, 255, 200))
        surface.blit(text_surf, (legend_x + 10, y_offset))
        y_offset += 18

    if has_path_items:
        y_offset += 4
        path_parts = []
        if path_summary.get("keys", 0) > 0:
            path_parts.append(f"{path_summary['keys']}[K]")
        if path_summary.get("boss_keys", 0) > 0:
            path_parts.append(f"{path_summary['boss_keys']}[BK]")
        if path_summary.get("ladders", 0) > 0:
            path_parts.append(f"{path_summary['ladders']}[L]")
        if path_summary.get("doors_locked", 0) > 0:
            path_parts.append(f"{path_summary['doors_locked']}[D]")
        if path_summary.get("doors_bomb", 0) > 0:
            path_parts.append(f"{path_summary['doors_bomb']}[Bx]")
        if path_summary.get("doors_boss", 0) > 0:
            path_parts.append(f"{path_summary['doors_boss']}[BD]")

        if path_parts:
            path_text = f"Path ahead: {' '.join(path_parts)}"
            path_surf = gui.small_font.render(path_text, True, (100, 255, 150))
            surface.blit(path_surf, (legend_x + 10, y_offset))
