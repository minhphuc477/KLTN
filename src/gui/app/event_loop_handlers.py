"""Compatibility facade for decomposed event handlers.

This module intentionally re-exports the public handler functions so existing
imports (for example in gui_runner.py) remain stable while implementation is
split across focused modules under src.gui.app.event_handlers.
"""

from __future__ import annotations

from src.gui.app.event_handlers import (
    clear_stale_preview_overlay,
    handle_global_keydown_shortcuts,
    handle_keydown_event,
    handle_keyup_event,
    handle_mouse_button_down_event,
    handle_mouse_button_down_preamble,
    handle_mouse_button_up_event,
    handle_mouse_motion_diagnostics,
    handle_mouse_motion_event,
    handle_mousewheel_event,
    handle_preview_overlay_events,
    handle_videoresize_event,
    handle_window_focus_event,
    poll_pygame_events,
    run_input_focus_fallback,
)

__all__ = [
    "clear_stale_preview_overlay",
    "handle_global_keydown_shortcuts",
    "handle_keydown_event",
    "handle_keyup_event",
    "handle_mouse_button_down_event",
    "handle_mouse_button_down_preamble",
    "handle_mouse_button_up_event",
    "handle_mouse_motion_diagnostics",
    "handle_mouse_motion_event",
    "handle_mousewheel_event",
    "handle_preview_overlay_events",
    "handle_videoresize_event",
    "handle_window_focus_event",
    "poll_pygame_events",
    "run_input_focus_fallback",
]
