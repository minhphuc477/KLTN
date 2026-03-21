from types import SimpleNamespace

from src.gui.controls.window_focus import force_focus, toggle_fullscreen


class _Screen:
    def __init__(self, size=(640, 480)):
        self._size = size

    def get_size(self):
        return self._size


class _Display:
    def __init__(self):
        self._screen = _Screen((640, 480))

    def Info(self):
        return SimpleNamespace(current_w=1280, current_h=720)

    def set_mode(self, size, flags=0):
        _ = flags
        self._screen = _Screen(size)
        return self._screen

    def flip(self):
        return None

    def get_wm_info(self):
        return {"window": 0}


class _Event:
    def set_grab(self, value):
        _ = value

    def pump(self):
        return None


class _Pygame:
    RESIZABLE = 1
    FULLSCREEN = 2
    HWSURFACE = 4
    DOUBLEBUF = 8

    def __init__(self):
        self.display = _Display()
        self.event = _Event()


def _make_logger():
    return SimpleNamespace(debug=lambda *a, **k: None, exception=lambda *a, **k: None)


def test_force_focus_noop_on_non_windows():
    gui = SimpleNamespace()
    pygame = _Pygame()
    logger = _make_logger()
    os_module = SimpleNamespace(name="posix")

    ok = force_focus(gui, pygame, logger, os_module)

    assert ok is False


def test_toggle_fullscreen_switches_and_sets_screen():
    gui = SimpleNamespace(
        fullscreen=False,
        screen_w=800,
        screen_h=600,
        _prev_window_size=(800, 600),
        screen=_Screen((800, 600)),
        control_panel_enabled=False,
    )
    gui._safe_set_mode = lambda size, flags=0: _Screen(size)
    gui._set_message = lambda msg, dur=0: None
    gui._show_toast = lambda msg, dur, typ: None
    gui._load_assets = lambda: None
    gui._render = lambda: None
    gui._center_view = lambda: None
    gui._update_control_panel_positions = lambda: None

    pygame = _Pygame()
    logger = _make_logger()
    os_module = SimpleNamespace(name="posix", environ={}, path=SimpleNamespace())
    platform_module = SimpleNamespace(system=lambda: "Linux")

    toggle_fullscreen(gui, pygame, logger, os_module, platform_module)

    assert gui.fullscreen is True
    assert gui.screen_w > 0 and gui.screen_h > 0

