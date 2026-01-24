import pytest
import numpy as np
from gui_runner import ToastNotification, PYGAME_AVAILABLE


@pytest.mark.skipif(not PYGAME_AVAILABLE, reason="Pygame not available")
def test_toast_render_with_various_alpha_types():
    import pygame
    pygame.init()
    surf = pygame.Surface((300, 80), pygame.SRCALPHA)

    t = ToastNotification("hello")

    # Normal int alpha
    t.get_alpha = lambda: 128
    t.render(surf, 150, 10)

    # Numpy int
    t.get_alpha = lambda: np.int64(128)
    t.render(surf, 150, 10)

    # Numpy float
    t.get_alpha = lambda: np.float64(128.0)
    t.render(surf, 150, 10)

    # Invalid string -> should be handled gracefully (fallback to opaque draw)
    t.get_alpha = lambda: "invalid"
    t.render(surf, 150, 10)

    pygame.quit()