"""Helpers for gui_runner event loop orchestration."""

from __future__ import annotations


def poll_pygame_events(pygame_module, time_module, logger):
    """Fetch pending pygame events with slow-call diagnostics."""
    try:
        start_ts = time_module.time()
        events = pygame_module.event.get()
        duration = time_module.time() - start_ts
        if duration > 0.05:
            logger.debug("Slow event.get() detected: %.3fs", duration)
        return events
    except Exception:
        logger.exception("pygame.event.get() raised")
        return []


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
            except Exception:
                logger.debug("Failed to clear event grab during fallback")
            try:
                pygame_module.mouse.set_visible(True)
            except Exception:
                logger.debug("Failed to set mouse visible during fallback")
            gui._last_ungrab_attempt = now_ts
    except Exception:
        logger.exception("Error during input focus fallback")


def clear_stale_preview_overlay(gui, logger):
    """Clear stale non-modal preview overlay state that can block interaction."""
    if not getattr(gui, "preview_overlay_visible", False):
        return
    if getattr(gui, "path_preview_dialog", None) or getattr(gui, "auto_path", None):
        return

    try:
        logger.warning("Clearing stale preview_overlay_visible (no dialog/path present) to restore input")
        gui.preview_overlay_visible = False
        gui.path_preview_dialog = None
        try:
            gui._set_message("Cleared stale preview overlay", 1.5)
        except Exception:
            pass
    except Exception:
        logger.exception("Failed to clear stale preview overlay")


def handle_window_focus_event(gui, event, pygame_module, logger):
    """Handle focus gain/loss events; returns True when consumed."""
    if event.type == getattr(pygame_module, "WINDOWFOCUSGAINED", None):
        logger.debug("WINDOWFOCUSGAINED: clearing event grab and showing mouse cursor")
        try:
            pygame_module.event.set_grab(False)
        except Exception:
            logger.debug("Could not clear event grab on focus gained")
        try:
            pygame_module.mouse.set_visible(True)
        except Exception:
            logger.debug("Could not set mouse visible on focus gained")
        try:
            gui._set_message("Window focused", 1.5)
        except Exception:
            pass
        return True

    if event.type == getattr(pygame_module, "WINDOWFOCUSLOST", None):
        logger.debug("WINDOWFOCUSLOST: pausing input interactions")
        try:
            gui._set_message("Window lost focus", 1.5)
        except Exception:
            pass
        return True

    return False
