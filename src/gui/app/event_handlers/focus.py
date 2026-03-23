"""Window focus and focus-recovery event handlers."""

from __future__ import annotations


def run_input_focus_fallback(gui, pygame_module, time_module, logger, should_attempt_focus_fallback_fn):
    """Try to recover input focus in windowed mode when focus is lost."""
    try:
        focused = pygame_module.mouse.get_focused()
        now_ts = time_module.time()
        if should_attempt_focus_fallback_fn(
            gui.fullscreen,
            focused,
            now_ts,
            getattr(gui, "_last_ungrab_attempt", 0.0),
            cooldown_sec=2.0,
        ):
            logger.debug("Window lacks input focus; attempting to clear event grab and show cursor")
            try:
                pygame_module.event.set_grab(False)
            except Exception as exc:
                logger.debug("Failed to clear event grab during fallback")
            try:
                pygame_module.mouse.set_visible(True)
            except Exception as exc:
                logger.debug("Failed to set mouse visible during fallback")
            gui._last_ungrab_attempt = now_ts
    except Exception as exc:
        logger.exception("Error during input focus fallback")


def handle_window_focus_event(gui, event, pygame_module, logger):
    """Handle focus gain/loss events; returns True when consumed."""
    if event.type == getattr(pygame_module, "WINDOWFOCUSGAINED", None):
        logger.debug("WINDOWFOCUSGAINED: clearing event grab and showing mouse cursor")
        try:
            pygame_module.event.set_grab(False)
        except Exception as exc:
            logger.debug("Could not clear event grab on focus gained")
        try:
            pygame_module.mouse.set_visible(True)
        except Exception as exc:
            logger.debug("Could not set mouse visible on focus gained")
        try:
            gui._set_message("Window focused", 1.5)
        except Exception as exc:
            pass
        return True

    if event.type == getattr(pygame_module, "WINDOWFOCUSLOST", None):
        logger.debug("WINDOWFOCUSLOST: pausing input interactions")
        try:
            gui._set_message("Window lost focus", 1.5)
        except Exception as exc:
            pass
        return True

    return False
