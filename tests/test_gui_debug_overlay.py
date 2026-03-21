from types import SimpleNamespace

from src.gui.overlay.debug_overlay import render_debug_overlay


class _Rendered:
    pass


class _Font:
    def render(self, text, antialias, color):
        _ = (text, antialias, color)
        return _Rendered()


class _Surface:
    def __init__(self):
        self.blits = []

    def fill(self, color):
        _ = color

    def blit(self, surf, pos):
        self.blits.append((surf, pos))


class _Pygame:
    SRCALPHA = object()

    @staticmethod
    def Surface(size, flags=None):
        _ = (size, flags)
        return _Surface()

    class font:
        @staticmethod
        def SysFont(name, size):
            _ = (name, size)
            return _Font()

    class mouse:
        @staticmethod
        def get_pos():
            return (10, 20)

        @staticmethod
        def get_focused():
            return True

    class event:
        @staticmethod
        def get_grab():
            return False

    class key:
        @staticmethod
        def name(value):
            return str(value)

    class draw:
        @staticmethod
        def rect(surface, color, rect, width=0):
            _ = (surface, color, rect, width)


def test_render_debug_overlay_runs_with_widget_data():
    surface = _Surface()
    widget = SimpleNamespace(control_name="w1", rect=(1, 2, 3, 4), is_open=False, state={})
    gui = SimpleNamespace(
        widget_manager=SimpleNamespace(widgets=[widget]),
        control_panel_rect=(0, 0, 100, 100),
        collapse_button_rect=(5, 5, 10, 10),
        _last_mouse_event={"type": "down", "pos": (1, 2), "button": 1, "time": 10.0},
        _last_key_event={"key": 65, "mods": 0, "time": 10.0},
        debug_click_log=[((1, 2), 9.0)],
    )
    time_module = SimpleNamespace(time=lambda: 11.0)

    render_debug_overlay(gui, surface, _Pygame, time_module)

    assert len(surface.blits) > 0

