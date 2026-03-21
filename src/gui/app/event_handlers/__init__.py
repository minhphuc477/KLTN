"""Decomposed event-loop handlers used by gui_runner."""

from __future__ import annotations

from .common import clear_stale_preview_overlay, poll_pygame_events
from .focus import handle_window_focus_event, run_input_focus_fallback
from .keyboard import handle_global_keydown_shortcuts, handle_keydown_event, handle_keyup_event
from .mouse import (
    handle_mouse_button_down_event,
    handle_mouse_button_down_preamble,
    handle_mouse_button_up_event,
    handle_mouse_motion_diagnostics,
    handle_mouse_motion_event,
    handle_mousewheel_event,
)
from .preview import handle_preview_overlay_events
from .window import handle_videoresize_event

__all__ = [
    "clear_stale_preview_overlay",
    "poll_pygame_events",
    "handle_window_focus_event",
    "run_input_focus_fallback",
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
]