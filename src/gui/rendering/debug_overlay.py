"""Debug overlay rendering helpers."""

from typing import Any


def render_debug_overlay(gui: Any, surface: Any, pygame: Any, time_module: Any) -> None:
    """Render debug overlay with mouse/widget diagnostics and recent clicks."""
    try:
        font = pygame.font.SysFont("Arial", 12)
    except Exception:
        return

    widget_count = len(gui.widget_manager.widgets) if gui.widget_manager else 0
    box_w = 380
    box_h = 24 + 16 * min(10, widget_count)
    box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    box_surf.fill((20, 20, 30, 220))
    surface.blit(box_surf, (10, 10))

    x = 14
    y = 14
    mouse_pos = pygame.mouse.get_pos()
    surface.blit(font.render(f"Mouse: {mouse_pos}", True, (220, 220, 220)), (x, y))
    y += 16
    panel_rect = getattr(gui, "control_panel_rect", None)
    surface.blit(font.render(f"Panel rect: {panel_rect}", True, (220, 220, 220)), (x, y))
    y += 16
    collapse_rect = getattr(gui, "collapse_button_rect", None)
    surface.blit(font.render(f"Collapse btn: {collapse_rect}", True, (220, 220, 220)), (x, y))
    y += 18

    focused = pygame.mouse.get_focused()
    grabbed = pygame.event.get_grab()
    now = time_module.time()
    surface.blit(font.render(f"Focused: {focused}   Grabbed: {grabbed}", True, (200, 240, 200)), (x, y))
    y += 16

    last = getattr(gui, "_last_mouse_event", None)
    if last:
        age = int((now - last.get("time", now)) * 1000)
        ltxt = f"Last mouse: {last.get('type')} pos={last.get('pos')} btn={last.get('button', '')} age={age}ms"
        surface.blit(font.render(ltxt, True, (200, 240, 200)), (x, y))
        y += 16

    lastk = getattr(gui, "_last_key_event", None)
    if lastk:
        k_age = int((now - lastk.get("time", now)) * 1000)
        try:
            kname = pygame.key.name(lastk.get("key"))
        except Exception:
            kname = str(lastk.get("key"))
        ktxt = f"Last key: {kname} age={k_age}ms mods={lastk.get('mods')}"
        surface.blit(font.render(ktxt, True, (200, 240, 200)), (x, y))
        y += 16

    if gui.widget_manager:
        for widget in gui.widget_manager.widgets[:8]:
            info = (
                f"{getattr(widget, 'control_name', widget.__class__.__name__)} "
                f"rect={widget.rect} open={getattr(widget, 'is_open', False)} state={widget.state}"
            )
            surface.blit(font.render(info, True, (200, 200, 255)), (x, y))
            y += 14

    if panel_rect:
        try:
            pygame.draw.rect(surface, (200, 80, 80), panel_rect, 2)
        except Exception:
            pass

    if collapse_rect:
        try:
            pygame.draw.rect(surface, (80, 200, 120), collapse_rect, 2)
        except Exception:
            pass

    cx = 14
    cy = box_h + 30
    surface.blit(font.render("Recent clicks (latest first):", True, (200, 200, 180)), (cx, cy))
    cy += 14
    for pos, ts in (gui.debug_click_log[:8] if getattr(gui, "debug_click_log", None) else []):
        surface.blit(font.render(f"{pos} @ {int(ts)}", True, (220, 220, 180)), (cx, cy))
        cy += 12
