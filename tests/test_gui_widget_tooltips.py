from types import SimpleNamespace

from src.gui.common.widget_tooltips import draw_tooltip, render_tooltips


class _Rect:
    def __init__(self, x=0, y=0, w=20, h=20):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.topleft = (x, y)

    def collidepoint(self, pos):
        px, py = pos
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


class _Rendered:
    def get_rect(self, **kwargs):
        return _Rect(kwargs.get("x", 0), kwargs.get("y", 0), kwargs.get("width", 40), kwargs.get("height", 20))


class _Font:
    def render(self, text, antialias, color):
        _ = (text, antialias, color)
        return _Rendered()

    def get_height(self):
        return 12


class _Surface:
    def __init__(self):
        self.blits = []
        self.fills = []

    def fill(self, color):
        self.fills.append(color)

    def blit(self, surf, pos):
        self.blits.append((surf, pos))


class _Pygame:
    SRCALPHA = object()

    @staticmethod
    def Rect(x, y, w, h):
        return _Rect(x, y, w, h)

    @staticmethod
    def Surface(size, flags=None):
        _ = (size, flags)
        return _Surface()

    class font:
        @staticmethod
        def SysFont(name, size):
            _ = (name, size)
            return _Font()

    class draw:
        @staticmethod
        def rect(surface, color, rect, width=0):
            _ = (surface, color, rect, width)


class _ButtonWidget:
    def __init__(self, label):
        self.label = label
        self.rect = _Rect(20, 40, 100, 24)
        self.state = {}


def _make_gui_with_widget(widget):
    gui = SimpleNamespace()
    gui.widget_manager = SimpleNamespace(widgets=[widget])
    gui.font = _Font()
    gui.control_panel_can_scroll = False
    gui.control_panel_scroll = 0
    gui.control_panel_rect = _Rect(0, 0, 200, 200)
    gui.screen_w = 320
    gui.screen_h = 240
    return gui


def test_draw_tooltip_renders_box_and_text():
    gui = SimpleNamespace(screen_w=320, screen_h=240)
    surface = _Surface()

    draw_tooltip(gui, surface, (100, 100), "hello", _Pygame)

    assert len(surface.blits) >= 2


def test_render_tooltips_for_button_label():
    widget = _ButtonWidget("Start Auto-Solve")
    gui = _make_gui_with_widget(widget)
    surface = _Surface()

    render_tooltips(gui, surface, (30, 45), _ButtonWidget, _Pygame)

    assert len(surface.blits) > 0

