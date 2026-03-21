"""Control panel scroll behavior helpers."""

from typing import Any


def update_control_panel_scroll(gui: Any, time_module: Any) -> None:
    """Apply momentum-based control panel scrolling with clamped bounds."""
    if not getattr(gui, "control_panel_can_scroll", False):
        return
    if getattr(gui, "control_panel_scroll_dragging", False):
        return

    vel = getattr(gui, "control_panel_scroll_velocity", 0.0)
    if abs(vel) < 1.0:
        gui.control_panel_scroll_velocity = 0.0
        return

    prev = gui.control_panel_scroll
    gui.control_panel_scroll = max(
        0,
        min(
            getattr(gui, "control_panel_scroll_max", 0),
            gui.control_panel_scroll + vel * max(1e-6, gui.delta_time),
        ),
    )

    if gui.control_panel_scroll <= 0 or gui.control_panel_scroll >= getattr(gui, "control_panel_scroll_max", 0):
        gui.control_panel_scroll_velocity = 0.0
    else:
        damping = getattr(gui, "control_panel_scroll_damping", 6.0)
        gui.control_panel_scroll_velocity *= max(0.0, 1.0 - damping * gui.delta_time)
        if abs(gui.control_panel_scroll_velocity) < 1.0:
            gui.control_panel_scroll_velocity = 0.0

    if abs(gui.control_panel_scroll - prev) > 0.5:
        gui.control_panel_ignore_click_until = time_module.time() + 0.12
