"""Display lifecycle and recovery helpers for pygame window management."""

from typing import Any


def safe_set_mode(size: tuple, pygame: Any, logger: Any, flags: int = 0, allow_fallback: bool = True):
    """Robust wrapper around pygame.display.set_mode with fallback behavior."""
    try:
        screen = pygame.display.set_mode(size, flags)
    except Exception:
        logger.exception("set_mode(%s, flags=%s) failed; attempting display reinit", size, flags)
        try:
            pygame.display.quit()
            pygame.display.init()
            screen = pygame.display.set_mode(size, flags)
        except Exception:
            logger.exception("Reinit + set_mode failed")
            screen = None

    try:
        if screen is None:
            raise RuntimeError("No screen")
        w, h = screen.get_size()
        if w == 0 or h == 0:
            raise RuntimeError(f"Invalid screen size {w}x{h}")
        return screen
    except Exception:
        logger.exception("Created screen is invalid")
        if not allow_fallback:
            return None
        try:
            logger.warning("Falling back to windowed 800x600")
            return pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        except Exception:
            logger.exception("Fallback windowed mode failed")
            return None


def attempt_display_reinit(gui: Any, pygame: Any, logger: Any) -> bool:
    """Fully reinitialize display and restore fullscreen/windowed mode."""
    try:
        pygame.display.quit()
        pygame.display.init()
        if getattr(gui, "fullscreen", False):
            try:
                disp = pygame.display.Info()
                new_size = (int(disp.current_w), int(disp.current_h))
            except Exception:
                new_size = (800, 600)
            flags = pygame.FULLSCREEN | getattr(pygame, "HWSURFACE", 0) | getattr(pygame, "DOUBLEBUF", 0)
            screen = safe_set_mode(new_size, pygame, logger, flags)
        else:
            prev = getattr(gui, "_prev_window_size", (800, 600))
            screen = safe_set_mode(prev, pygame, logger, pygame.RESIZABLE)

        if screen:
            gui.screen = screen
            try:
                gui.screen_w, gui.screen_h = gui.screen.get_size()
            except Exception:
                gui.screen_w, gui.screen_h = (800, 600)
            try:
                gui._load_assets()
            except Exception:
                logger.exception("Failed to reload assets after display reinit")
            return True
    except Exception:
        logger.exception("Display reinit failed")
    return False


def ensure_display_alive(gui: Any, pygame: Any, logger: Any, force: bool = False) -> bool:
    """Check display health and attempt recovery if needed."""
    _ = force
    try:
        surf = pygame.display.get_surface()
        if surf is None:
            logger.warning("Display surface is None; attempting recovery")
            recovered = attempt_display_reinit(gui, pygame, logger)
            if recovered:
                gui._show_toast("Recovered display", 3.0, "warning")
                logger.info("Display recovered successfully")
                gui._display_recovery_attempts = 0
                return True
            gui._show_toast("Display unavailable; fallback failed", 5.0, "error")
            return False

        w, h = surf.get_size()
        if w == 0 or h == 0:
            logger.warning("Display surface has invalid size %sx%s; attempting recovery", w, h)
            recovered = attempt_display_reinit(gui, pygame, logger)
            if recovered:
                gui._show_toast("Recovered display", 3.0, "warning")
                gui._display_recovery_attempts = 0
                return True
            gui._show_toast("Display invalid; returning to windowed mode", 5.0, "error")
            return False

        gui._display_recovery_attempts = 0
        return True
    except Exception:
        logger.exception("Error while checking display health")
        return False
