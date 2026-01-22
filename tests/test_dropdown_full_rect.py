import pygame
from src.gui.widgets import DropdownWidget


def test_dropdown_full_rect_includes_label():
    pygame.init()
    d = DropdownWidget((20, 40), 'Label', ['a','b'], selected=0)
    assert hasattr(d, 'full_rect')
    assert d.full_rect.y < d.rect.y
    assert d.full_rect.height > d.rect.height
    pygame.quit()
