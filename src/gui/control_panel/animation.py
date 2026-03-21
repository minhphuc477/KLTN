"""Control panel animation and scrolling helpers."""

from typing import Any


def start_toggle_panel_animation(gui: Any, target_collapsed: bool, time_module: Any) -> None:
    """Begin animated transition to collapsed or expanded state."""
    collapsed_width = 40
    target_width = float(collapsed_width) if target_collapsed else float(gui.control_panel_width)
    gui.control_panel_anim_from = float(gui.control_panel_width_current)
    gui.control_panel_anim_to = float(target_width)
    gui.control_panel_anim_start = time_module.time()
    gui.control_panel_anim_duration = 0.22
    gui.control_panel_target_collapsed = target_collapsed
    gui.control_panel_animating = True


def update_control_panel_animation(gui: Any, time_module: Any) -> None:
    """Advance control panel animation state one frame."""
    if not getattr(gui, "control_panel_animating", False):
        return

    try:
        gui._update_control_panel_positions()
    except Exception:
        pass

    elapsed = time_module.time() - gui.control_panel_anim_start
    t = min(1.0, elapsed / max(1e-6, gui.control_panel_anim_duration))
    ease = t * t * (3 - 2 * t)
    gui.control_panel_width_current = gui.control_panel_anim_from + (gui.control_panel_anim_to - gui.control_panel_anim_from) * ease

    if t >= 1.0:
        gui.control_panel_animating = False
        gui.control_panel_width_current = gui.control_panel_anim_to
        gui.control_panel_collapsed = bool(gui.control_panel_target_collapsed)
        try:
            if not gui.control_panel_collapsed:
                gui._update_control_panel_positions()
            gui._set_message(f"Panel: {'collapsed' if gui.control_panel_collapsed else 'expanded'}")
        except Exception:
            pass
