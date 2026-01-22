import pygame
from src.gui.widgets import DropdownWidget


def test_dropdown_persist_flag():
    pygame.init()
    d = DropdownWidget((0, 0), 'Test', ['A', 'B', 'C'], selected=0, keep_open_on_select=True)
    assert not d.is_open
    # simulate opening
    d.is_open = True
    # simulate clicking option 1
    handled = d.handle_mouse_down((d.dropdown_rect.x + 5, d.dropdown_rect.y + 24 * 1 + 2), 1)
    assert handled is True
    # Because keep_open_on_select is True, the dropdown should remain open
    assert d.is_open is True
    pygame.quit()


def test_dropdown_default_closes_on_select():
    pygame.init()
    d = DropdownWidget((0, 0), 'Test', ['A', 'B', 'C'], selected=0)
    d.is_open = True
    handled = d.handle_mouse_down((d.dropdown_rect.x + 5, d.dropdown_rect.y + 2), 1)
    assert handled is True
    assert d.is_open is False
    pygame.quit()
