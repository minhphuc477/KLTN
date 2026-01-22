import pygame
from src.gui.widgets import DropdownWidget, WidgetManager


def test_dropdown_state_snapshot_and_restore():
    pygame.init()
    wm = WidgetManager()
    d1 = DropdownWidget((0,0),'D', ['a','b','c'], selected=1)
    d1.control_name = 'd'
    d1.is_open = True
    wm.add_widget(d1)

    snap = wm.snapshot_dropdown_state()
    assert 'd' in snap and snap['d']['is_open'] is True and snap['d']['selected'] == 1

    # Simulate rebuild
    wm.widgets.clear()
    d2 = DropdownWidget((0,0),'D', ['a','b','c'], selected=0)
    d2.control_name = 'd'
    wm.add_widget(d2)

    wm.apply_dropdown_state(snap)
    assert any(w.control_name == 'd' and w.is_open for w in wm.widgets)
    pygame.quit()
