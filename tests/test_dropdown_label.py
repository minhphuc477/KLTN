import pygame
from src.gui.widgets import DropdownWidget


def test_dropdown_renders_label():
    pygame.init()
    surf = pygame.Surface((300, 200))
    d = DropdownWidget((20, 40), 'Threshold', ['0.70','0.75','0.80'], selected=0)
    # render should not raise and should draw label above
    d.render(surf)
    pygame.quit()
