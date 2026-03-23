"""Shared event-loop utilities for polling and generic maintenance."""

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
    except Exception as exc:
        logger.exception("pygame.event.get() raised")
        return []


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
        except Exception as exc:
            pass
    except Exception as exc:
        logger.exception("Failed to clear stale preview overlay")
