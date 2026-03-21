"""Helpers for rendering the full-screen help overlay."""

from typing import Any


def render_help_overlay(gui: Any, pygame: Any) -> None:
    """Render help overlay with sections and control hints."""
    overlay = pygame.Surface((gui.screen_w, gui.screen_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    gui.screen.blit(overlay, (0, 0))

    help_lines = [
        "ZAVE - Zelda AI Validation Environment",
        "",
        "Movement:",
        "  Arrow Keys - Move Link",
        "  Mouse Wheel - Zoom in/out",
        "  Middle Mouse Drag - Pan view",
        "",
        "Actions:",
        "  SPACE - Run A* auto-solver",
        "  R - Reset current map",
        "  N - Next map",
        "  P - Previous map",
        "",
        "View:",
        "  +/- or Wheel - Zoom in/out",
        "  0 - Reset zoom to default",
        "  C - Center view on player",
        "  M - Toggle minimap",
        "  H - Toggle A* heatmap",
        "  TAB - Toggle control panel",
        "  F11 - Toggle fullscreen",
        "",
        "Speed Control:",
        "  [ or , - Decrease speed",
        "  ] or . - Increase speed",
        "  (Speeds: 0.25x, 0.5x, 1x, 2x, 5x, 10x)",
        "",
        "Press F1 or ESC to close this help",
    ]

    y = 50
    for line in help_lines:
        if line.startswith("ZAVE"):
            surf = gui.big_font.render(line, True, (100, 200, 255))
        elif line.endswith(":") and not line.startswith(" "):
            surf = gui.font.render(line, True, (255, 200, 100))
        else:
            surf = gui.small_font.render(line, True, (200, 200, 200))
        gui.screen.blit(surf, (50, y))
        y += 22 if line else 10
