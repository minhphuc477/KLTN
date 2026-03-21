"""Helpers for status message and toast notification state."""

from typing import Any


def set_message(gui: Any, message: str, duration: float = 3.0, time_module: Any = None) -> None:
    """Set status message with timestamp for auto-hide."""
    tm = time_module if time_module is not None else __import__("time")
    gui.message = message
    gui.message_time = tm.time()
    gui.message_duration = duration


def show_toast(
    gui: Any,
    message: str,
    duration: float = 3.0,
    toast_type: str = "info",
    toast_notification_cls: Any = None,
) -> None:
    """Append a floating toast notification if supported by GUI state."""
    if not hasattr(gui, "toast_notifications"):
        return
    if toast_notification_cls is None:
        return
    gui.toast_notifications.append(toast_notification_cls(message, duration, toast_type))


def update_toasts(gui: Any) -> None:
    """Update and remove expired toast notifications."""
    if hasattr(gui, "toast_notifications"):
        gui.toast_notifications = [t for t in gui.toast_notifications if not t.is_expired()]


def render_toasts(gui: Any, surface: Any) -> None:
    """Render active toast notifications."""
    if not hasattr(gui, "toast_notifications"):
        return
    base_y = gui.screen_h - 140
    for i, toast in enumerate(gui.toast_notifications):
        toast.render(surface, gui.screen_w // 2, base_y - i * 70)
