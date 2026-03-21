from types import SimpleNamespace

from src.gui.controls.control_panel_scroll import update_control_panel_scroll


def _make_gui():
    return SimpleNamespace(
        control_panel_can_scroll=True,
        control_panel_scroll_dragging=False,
        control_panel_scroll_velocity=120.0,
        control_panel_scroll=10.0,
        control_panel_scroll_max=200.0,
        control_panel_scroll_damping=6.0,
        control_panel_ignore_click_until=0.0,
        delta_time=0.1,
    )


def test_update_control_panel_scroll_applies_motion_and_damping():
    gui = _make_gui()
    time_module = SimpleNamespace(time=lambda: 5.0)

    update_control_panel_scroll(gui, time_module)

    assert gui.control_panel_scroll > 10.0
    assert gui.control_panel_scroll_velocity < 120.0
    assert gui.control_panel_ignore_click_until >= 5.0


def test_update_control_panel_scroll_stops_near_zero_velocity():
    gui = _make_gui()
    gui.control_panel_scroll_velocity = 0.2
    time_module = SimpleNamespace(time=lambda: 5.0)

    update_control_panel_scroll(gui, time_module)

    assert gui.control_panel_scroll_velocity == 0.0

