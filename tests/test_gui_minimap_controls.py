import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.gui.controls.minimap_controls import handle_minimap_click, render_minimap


class DummyState:
    def __init__(self):
        self.position = (1, 1)
        self.collected_items = set()


class DummyEnv:
    def __init__(self):
        self.height = 4
        self.width = 4
        self.grid = np.full((4, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int64)
        self.grid[0, 0] = SEMANTIC_PALETTE["START"]
        self.state = DummyState()

    def _find_all_positions(self, tile_id):
        positions = []
        ys, xs = np.where(self.grid == tile_id)
        for y, x in zip(ys, xs):
            positions.append((int(y), int(x)))
        return positions


class FakeRect:
    pass


class FakeSurface:
    def __init__(self, size):
        self.size = size
        self.blit_calls = []

    def get_rect(self):
        return FakeRect()

    def blit(self, obj, pos):
        self.blit_calls.append((obj, pos))


class FakeFont:
    def render(self, text, aa, color):
        return (text, aa, color)


class FakeDraw:
    def rect(self, *args, **kwargs):
        return None

    def circle(self, *args, **kwargs):
        return None


class FakePygame:
    SRCALPHA = 0

    def __init__(self):
        self.draw = FakeDraw()
        self.font = type("F", (), {"SysFont": lambda *a, **k: FakeFont()})

    def Surface(self, size, _flags=None):
        return FakeSurface(size)


class DummyGui:
    def __init__(self):
        self.env = DummyEnv()
        self.show_minimap = True
        self.screen_w = 1000
        self.screen_h = 700
        self.SIDEBAR_WIDTH = 260
        self.HUD_HEIGHT = 140
        self.minimap_size = 120
        self.screen = FakeSurface((1000, 700))
        self.TILE_SIZE = 32
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.message = ""

    def _clamp_view_offset(self):
        return None


def test_handle_minimap_click_outside_returns_false():
    gui = DummyGui()
    assert handle_minimap_click(gui, (0, 0)) is False


def test_handle_minimap_click_inside_updates_view_and_message():
    gui = DummyGui()
    x = gui.screen_w - gui.SIDEBAR_WIDTH - gui.minimap_size + 20
    y = gui.screen_h - gui.HUD_HEIGHT - gui.minimap_size + 20
    assert handle_minimap_click(gui, (x, y)) is True
    assert gui.message.startswith("Jumped to")


def test_render_minimap_executes_with_fake_pygame():
    gui = DummyGui()
    render_minimap(gui, FakePygame())
    assert len(gui.screen.blit_calls) >= 1

