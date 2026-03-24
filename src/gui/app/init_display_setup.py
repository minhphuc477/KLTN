"""Display/window setup helpers for ZeldaGUI initialization."""

from __future__ import annotations

from typing import Any


def initialize_display_window(*, gui: Any, pygame: Any, os_module: Any, logger: Any) -> None:
    """Configure display sizing, create the window, and apply focus helpers."""
    gui.zoom_idx = gui.DEFAULT_ZOOM_IDX
    gui.TILE_SIZE = gui.ZOOM_LEVELS[gui.zoom_idx]
    gui.HUD_HEIGHT = 10
    gui.SIDEBAR_WIDTH = 220

    display_info = pygame.display.Info()
    max_screen_w = display_info.current_w - 100
    max_screen_h = display_info.current_h - 100

    def grid_shape(m: Any) -> tuple[int, int]:
        grid = getattr(m, "global_grid", m)
        return getattr(grid, "shape")[0], getattr(grid, "shape")[1]

    max_map_h = max(grid_shape(m)[0] for m in gui.maps)
    max_map_w = max(grid_shape(m)[1] for m in gui.maps)

    ideal_w = max_map_w * gui.TILE_SIZE + gui.SIDEBAR_WIDTH
    ideal_h = max_map_h * gui.TILE_SIZE + gui.HUD_HEIGHT

    gui.screen_w = min(ideal_w, max_screen_w)
    gui.screen_h = min(ideal_h, max_screen_h)
    gui.screen_w = max(gui.screen_w, gui.MIN_WIDTH)
    gui.screen_h = max(gui.screen_h, gui.MIN_HEIGHT)

    gui.screen = pygame.display.set_mode((gui.screen_w, gui.screen_h), pygame.RESIZABLE)
    pygame.display.set_caption("ZAVE: Zelda AI Validation Environment")
    gui._prev_window_size = (gui.screen_w, gui.screen_h)

    try:
        pygame.event.set_grab(False)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.debug("Could not clear event grab at startup")
    try:
        pygame.mouse.set_visible(True)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.debug("Could not ensure mouse cursor visible at startup")

    try:
        if os_module.name == "nt":
            try:
                import ctypes

                user32 = ctypes.windll.user32
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    logger.debug("Attempting to bring window to foreground (hwnd=%s)", hwnd)
                    user32.ShowWindow(hwnd, 5)
                    user32.SetForegroundWindow(hwnd)
                    pygame.event.pump()
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                    logger.debug("Set focus to window via Win32 API")
            except (AttributeError, RuntimeError, ValueError, TypeError):
                logger.debug("Windows focus helper failed", exc_info=True)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.debug("Focus bring-to-front helper encountered an error", exc_info=True)

