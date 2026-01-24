import os
import pygame
import pytest

from gui_runner import ZeldaGUI
from src.gui.widgets import CheckboxWidget, DropdownWidget


@pytest.fixture(autouse=True)
def init_pygame():
    # Use dummy video driver to avoid opening windows in CI
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.init()
    yield
    pygame.quit()


import numpy as np

def test_item_spacing_updated():
    # Small map to keep window small
    grid = np.zeros((10, 10), dtype=int)
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])
    # Force small screen to test layout
    gui.screen_h = 400
    gui._init_control_panel()
    # Ensure widgets exist
    assert gui.widget_manager and len(gui.widget_manager.widgets) > 2

    # Find first two checkbox widgets and compare their vertical spacing
    checkbox_positions = [w.pos for w in gui.widget_manager.widgets if isinstance(w, CheckboxWidget)]
    assert len(checkbox_positions) >= 2
    y0 = checkbox_positions[0][1]
    y1 = checkbox_positions[1][1]
    assert y1 - y0 == 44


def test_control_panel_scroll_allows_reach_bottom():
    import numpy as np
    grid = np.zeros((10, 10), dtype=int)
    gui = ZeldaGUI(maps=[grid], map_names=['M1'])
    # Make screen very short to force scrolling
    gui.screen_h = 220
    gui._init_control_panel()
    # Recompute positions with small screen
    gui._update_control_panel_positions()
    assert getattr(gui, 'control_panel_can_scroll', False) is True

    # Compute actual bottom-most widget bottom
    panel_x = gui.control_panel_rect.x
    panel_y = gui.control_panel_rect.y
    max_bottom = 0
    for w in gui.widget_manager.widgets:
        # Consider full_rect if present
        if getattr(w, 'full_rect', None) is not None:
            bottom = w.full_rect.bottom
        else:
            bottom = w.rect.bottom
        if bottom > max_bottom:
            max_bottom = bottom

    # The scroll range plus panel height should be enough to reach the bottom-most widget
    panel_height = gui.control_panel_rect.height
    scroll_max = getattr(gui, 'control_panel_scroll_max', 0)

    assert panel_y + panel_height + scroll_max >= max_bottom
