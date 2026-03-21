import math
from types import SimpleNamespace

from src.gui.overlay.status_display import (
    render_error_banner,
    render_solver_status_banner,
    render_status_bar,
    show_error,
    show_message,
    show_warning,
)


class _Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.right = x + w


class _Rendered:
    def __init__(self):
        self.alpha = None

    def set_alpha(self, value):
        self.alpha = value

    def get_rect(self, **kwargs):
        return _Rect(
            kwargs.get("x", 0),
            kwargs.get("y", 0),
            kwargs.get("width", 10),
            kwargs.get("height", 10),
        )


class _Font:
    def render(self, text, antialias, color):
        _ = (text, antialias, color)
        return _Rendered()


class _Surface:
    def __init__(self, size=(0, 0), flags=None):
        self.size = size
        self.flags = flags
        self.fills = []
        self.blits = []

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
        return _Surface(size, flags)

    class font:
        @staticmethod
        def SysFont(name, size):
            _ = (name, size)
            return _Font()

    class draw:
        calls = []

        @classmethod
        def rect(cls, surface, color, rect, width):
            cls.calls.append((surface, color, rect, width))


class _Clock:
    def get_fps(self):
        return 59.9


def _make_gui():
    gui = SimpleNamespace()
    gui.screen_w = 640
    gui.screen_h = 480
    gui.status_message = "Ready"
    gui.auto_mode = False
    gui.auto_step_idx = 0
    gui.auto_path = []
    gui.clock = _Clock()
    gui.solver_running = False
    gui.algorithm_idx = 0
    gui.solver_algorithm_idx = 1
    gui.error_message = None
    gui.error_time = 0.0
    gui._algorithm_name = lambda idx: f"Alg{idx}"
    gui._get_path_items_display_text = lambda: "K1 B1"
    return gui


def test_show_error_sets_state_and_logs():
    gui = _make_gui()
    logs = []
    logger = SimpleNamespace(error=lambda msg: logs.append(("error", msg)))
    time_module = SimpleNamespace(time=lambda: 123.4)

    show_error(gui, "boom", logger, time_module)

    assert gui.error_message == "boom"
    assert gui.error_time == 123.4
    assert gui.status_message == "Error"
    assert logs == [("error", "boom")]


def test_show_message_sets_state_and_logs():
    gui = _make_gui()
    logs = []
    logger = SimpleNamespace(info=lambda msg: logs.append(("info", msg)))
    time_module = SimpleNamespace(time=lambda: 42.0)

    show_message(gui, "hello", 1.5, logger, time_module)

    assert gui.message == "hello"
    assert gui.message_time == 42.0
    assert gui.message_duration == 1.5
    assert gui.status_message == "Info"
    assert logs == [("info", "hello")]


def test_show_warning_prefixes_message_and_logs():
    gui = _make_gui()
    logs = []
    logger = SimpleNamespace(warning=lambda msg: logs.append(("warning", msg)))

    show_warning(gui, "careful", logger)

    assert gui.message == "[!] careful"
    assert logs == [("warning", "careful")]


def test_render_error_banner_clears_after_timeout():
    gui = _make_gui()
    gui.error_message = "oops"
    gui.error_time = 10.0
    surface = _Surface((640, 480))
    time_module = SimpleNamespace(time=lambda: 16.0)

    render_error_banner(gui, surface, _Pygame, time_module)

    assert gui.error_message is None


def test_render_solver_status_banner_renders_when_running():
    gui = _make_gui()
    gui.solver_running = True
    surface = _Surface((640, 480))
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    time_module = SimpleNamespace(time=lambda: 1.0)

    render_solver_status_banner(gui, surface, _Pygame, math, time_module, logger)

    assert len(surface.blits) > 0


def test_render_status_bar_renders_base_content():
    gui = _make_gui()
    surface = _Surface((640, 480))

    render_status_bar(gui, surface, _Pygame)

    assert len(surface.blits) > 0

