"""Helpers for status/error messaging and status banner rendering."""

from typing import Any


def show_error(gui: Any, message: str, logger: Any, time_module: Any) -> None:
    """Display error message state with timestamp and status label."""
    logger.error(message)
    gui.error_message = message
    gui.error_time = time_module.time()
    gui.status_message = "Error"


def show_message(gui: Any, message: str, duration: float, logger: Any, time_module: Any) -> None:
    """Display informational message state with expiration metadata."""
    logger.info(message)
    gui.message = message
    gui.message_time = time_module.time()
    gui.message_duration = duration
    gui.status_message = "Info"


def show_warning(gui: Any, message: str, logger: Any) -> None:
    """Display warning message to user and logs."""
    logger.warning(message)
    gui.message = f"[!] {message}"


def render_error_banner(gui: Any, surface: Any, pygame: Any, time_module: Any) -> None:
    """Render timed fade-out error banner at top of screen."""
    if not hasattr(gui, "error_message") or not gui.error_message:
        return
    elapsed = time_module.time() - gui.error_time
    if elapsed >= 5.0:
        gui.error_message = None
        return

    alpha = 1.0 if elapsed < 4.0 else (5.0 - elapsed)
    banner_height = 45
    banner_rect = pygame.Rect(0, 0, gui.screen_w, banner_height)
    banner_surface = pygame.Surface((gui.screen_w, banner_height), pygame.SRCALPHA)
    banner_surface.fill((200, 0, 0, int(220 * alpha)))
    surface.blit(banner_surface, (0, 0))

    font = pygame.font.SysFont("Arial", 28)
    text = f"[!] {gui.error_message}"
    text_surf = font.render(text, True, (255, 255, 255))
    text_surf.set_alpha(int(255 * alpha))
    text_rect = text_surf.get_rect(center=(gui.screen_w // 2, banner_height // 2))
    surface.blit(text_surf, text_rect)
    pygame.draw.rect(surface, (150, 0, 0), banner_rect, 2)


def render_solver_status_banner(
    gui: Any,
    surface: Any,
    pygame: Any,
    math_module: Any,
    time_module: Any,
    logger: Any,
) -> None:
    """Render animated solver-running banner with active algorithm name."""
    if not getattr(gui, "solver_running", False):
        return

    alg_idx = getattr(gui, "solver_algorithm_idx", getattr(gui, "algorithm_idx", 0))
    alg_name = gui._algorithm_name(alg_idx)
    logger.debug(
        "BANNER: Rendering solver banner with solver_algorithm_idx=%d, alg_name=%s",
        alg_idx,
        alg_name,
    )

    banner_height = 50
    banner_y = 50
    banner_rect = pygame.Rect(0, banner_y, gui.screen_w, banner_height)
    banner_surface = pygame.Surface((gui.screen_w, banner_height), pygame.SRCALPHA)

    pulse = (math_module.sin(time_module.time() * 3) + 1) / 2
    alpha = int(180 + 75 * pulse)
    banner_surface.fill((200, 150, 0, alpha))
    surface.blit(banner_surface, (0, banner_y))

    font = pygame.font.SysFont("Arial", 32)
    text = f"Computing path with {alg_name}..."
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=(gui.screen_w // 2, banner_y + banner_height // 2))
    surface.blit(text_surf, text_rect)
    pygame.draw.rect(surface, (255, 200, 0), banner_rect, 2)


def render_status_bar(gui: Any, surface: Any, pygame: Any) -> None:
    """Render bottom status bar with algorithm, progress, and FPS."""
    bar_height = 30
    bar_y = gui.screen_h - bar_height
    bar_rect = pygame.Rect(0, bar_y, gui.screen_w, bar_height)

    bar_surface = pygame.Surface((gui.screen_w, bar_height), pygame.SRCALPHA)
    bar_surface.fill((40, 40, 50, 200))
    surface.blit(bar_surface, (0, bar_y))

    font = pygame.font.SysFont("Arial", 20)

    status_text = f"Status: {gui.status_message}"
    status_surf = font.render(status_text, True, (180, 220, 255))
    surface.blit(status_surf, (10, bar_y + 7))

    alg_idx = getattr(gui, "algorithm_idx", 0)
    alg_name = gui._algorithm_name(alg_idx)
    alg_indicator = f"[{alg_name}]"
    alg_color = (255, 215, 0) if gui.auto_mode else (150, 180, 255)
    alg_surf = font.render(alg_indicator, True, alg_color)
    alg_rect = alg_surf.get_rect(center=(gui.screen_w // 2, bar_y + bar_height // 2))
    surface.blit(alg_surf, alg_rect)

    if gui.auto_mode:
        progress = f"Step {gui.auto_step_idx + 1}/{len(gui.auto_path)}"
        progress_surf = font.render(progress, True, (100, 255, 100))
        progress_rect = progress_surf.get_rect(centerx=alg_rect.right + 60, centery=bar_y + bar_height // 2)
        surface.blit(progress_surf, progress_rect)

        items_text = gui._get_path_items_display_text()
        if items_text:
            items_surf = font.render(items_text, True, (255, 220, 100))
            surface.blit(items_surf, (progress_rect.right + 20, bar_y + 7))
    elif getattr(gui, "auto_path", None) and len(gui.auto_path) > 0:
        items_text = gui._get_path_items_display_text()
        if items_text:
            preview_text = f"Path ready ({len(gui.auto_path)} steps) | {items_text}"
        else:
            preview_text = f"Path ready ({len(gui.auto_path)} steps) - Press ENTER to start"
        preview_surf = font.render(preview_text, True, (100, 200, 255))
        preview_rect = preview_surf.get_rect(center=(gui.screen_w // 2, bar_y + bar_height // 2))
        surface.blit(preview_surf, preview_rect)

    fps_text = f"FPS: {int(gui.clock.get_fps())}"
    fps_surf = font.render(fps_text, True, (255, 255, 180))
    fps_rect = fps_surf.get_rect(right=gui.screen_w - 10, centery=bar_y + bar_height // 2)
    surface.blit(fps_surf, fps_rect)

    pygame.draw.rect(surface, (60, 60, 80), bar_rect, 1)
