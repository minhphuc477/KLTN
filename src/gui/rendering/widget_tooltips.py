"""Widget tooltip rendering helpers for GUI controls."""

from typing import Any


def render_tooltips(gui: Any, surface: Any, mouse_pos: tuple, button_widget_cls: Any, pygame: Any) -> None:
    """Render tooltip for the first hovered widget with known tooltip mapping."""
    if not gui.widget_manager:
        return

    tooltips = {
        "show_heatmap": "Toggle A* search heatmap visualization",
        "show_minimap": "Toggle minimap display in top-right corner",
        "show_path": "Show/hide the solution path preview",
        "smooth_camera": "Enable smooth camera transitions",
        "show_grid": "Toggle grid overlay on map",
        "zoom": "Adjust map zoom level (also use +/- keys)",
        "difficulty": "Select map difficulty level",
        "algorithm": "Choose pathfinding algorithm for auto-solve",
        "ml_heuristic": "Use experimental ML-style heuristic (may be non-admissible)",
        "parallel_search": "Run multiple strategies in parallel and pick fastest result",
        "solver_comparison": "Run a comparison of available solvers and report metrics",
        "dstar_lite": "Enable D* Lite incremental replanning (if implemented)",
        "show_topology": "Draw room nodes & edges from topology graph on the map",
    }

    button_tooltips = {
        "Start Auto-Solve": "Begin automatic pathfinding solution (SPACE)",
        "Stop": "Stop the current auto-solve operation",
        "Generate Dungeon": "Create a new random dungeon map (BSP)",
        "AI Generate": "Generate dungeon using trained latent diffusion AI model",
        "Reset": "Reset current map to initial state (R key)",
        "Path Preview": "Preview the complete solution path",
        "Clear Path": "Clear the displayed path overlay",
        "Export Route": "Save current path to file",
        "Load Route": "Load path from file",
    }

    for widget in gui.widget_manager.widgets:
        test_pos = mouse_pos
        header_height = gui.font.get_height() + 12
        if (
            getattr(gui, "control_panel_can_scroll", False)
            and getattr(gui, "control_panel_rect", None)
            and gui.control_panel_rect.collidepoint(mouse_pos)
        ):
            panel_top = gui.control_panel_rect.y
            if mouse_pos[1] > panel_top + header_height:
                test_pos = (mouse_pos[0], mouse_pos[1] + getattr(gui, "control_panel_scroll", 0))

        if not hasattr(widget, "rect"):
            continue
        if not widget.rect.collidepoint(test_pos):
            continue
        if widget.rect.y < gui.control_panel_rect.y + header_height:
            continue

        tooltip_text = None
        if hasattr(widget, "flag_name") and widget.flag_name in tooltips:
            tooltip_text = tooltips[widget.flag_name]
        elif hasattr(widget, "control_name") and widget.control_name in tooltips:
            tooltip_text = tooltips[widget.control_name]
        elif isinstance(widget, button_widget_cls) and hasattr(widget, "label"):
            tooltip_text = button_tooltips.get(widget.label)

        if tooltip_text:
            draw_tooltip(gui, surface, mouse_pos, tooltip_text, pygame)
        break


def draw_tooltip(gui: Any, surface: Any, pos: tuple, text: str, pygame: Any) -> None:
    """Draw a tooltip box near cursor and keep it within screen bounds."""
    font = pygame.font.SysFont("Arial", 18)
    padding = 8

    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect()

    tooltip_x = pos[0] + 15
    tooltip_y = pos[1] + 15

    if tooltip_x + text_rect.width + padding * 2 > gui.screen_w:
        tooltip_x = pos[0] - text_rect.width - padding * 2 - 15
    if tooltip_y + text_rect.height + padding * 2 > gui.screen_h:
        tooltip_y = pos[1] - text_rect.height - padding * 2 - 15

    bg_rect = pygame.Rect(
        tooltip_x - padding,
        tooltip_y - padding,
        text_rect.width + padding * 2,
        text_rect.height + padding * 2,
    )
    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    bg_surface.fill((50, 50, 60, 240))
    surface.blit(bg_surface, bg_rect.topleft)

    pygame.draw.rect(surface, (100, 150, 200), bg_rect, 2)
    surface.blit(text_surf, (tooltip_x, tooltip_y))
