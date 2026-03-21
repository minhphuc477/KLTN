from types import SimpleNamespace

from src.gui.controls.control_panel_animation import (
    start_toggle_panel_animation,
    update_control_panel_animation,
)


def _make_gui():
    gui = SimpleNamespace()
    gui.control_panel_width = 320
    gui.control_panel_width_current = 320.0
    gui.control_panel_anim_from = 0.0
    gui.control_panel_anim_to = 0.0
    gui.control_panel_anim_start = 0.0
    gui.control_panel_anim_duration = 0.22
    gui.control_panel_target_collapsed = False
    gui.control_panel_animating = False
    gui.control_panel_collapsed = False
    gui._set_message_calls = []
    gui._set_message = lambda msg: gui._set_message_calls.append(msg)
    gui._update_control_panel_positions = lambda: None
    return gui


def test_start_toggle_panel_animation_sets_state():
    gui = _make_gui()
    time_module = SimpleNamespace(time=lambda: 10.0)

    start_toggle_panel_animation(gui, True, time_module)

    assert gui.control_panel_animating is True
    assert gui.control_panel_anim_to == 40.0
    assert gui.control_panel_anim_start == 10.0


def test_update_control_panel_animation_completes():
    gui = _make_gui()
    gui.control_panel_animating = True
    gui.control_panel_anim_from = 320.0
    gui.control_panel_anim_to = 40.0
    gui.control_panel_anim_start = 0.0
    gui.control_panel_anim_duration = 0.22
    gui.control_panel_target_collapsed = True
    time_module = SimpleNamespace(time=lambda: 1.0)

    update_control_panel_animation(gui, time_module)

    assert gui.control_panel_animating is False
    assert gui.control_panel_collapsed is True
    assert gui.control_panel_width_current == 40.0
    assert gui._set_message_calls

