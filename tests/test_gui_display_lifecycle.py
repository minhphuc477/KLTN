from types import SimpleNamespace

from src.gui.overlay.display_lifecycle import attempt_display_reinit, ensure_display_alive, safe_set_mode


class _Screen:
    def __init__(self, size=(640, 480)):
        self._size = size

    def get_size(self):
        return self._size


class _Display:
    def __init__(self):
        self.surface = _Screen((640, 480))

    def set_mode(self, size, flags=0):
        _ = flags
        self.surface = _Screen(size)
        return self.surface

    def quit(self):
        return None

    def init(self):
        return None

    def get_surface(self):
        return self.surface

    def Info(self):
        return SimpleNamespace(current_w=1280, current_h=720)


class _Pygame:
    RESIZABLE = 1
    FULLSCREEN = 2
    HWSURFACE = 4
    DOUBLEBUF = 8

    def __init__(self):
        self.display = _Display()


def _make_logger():
    return SimpleNamespace(
        exception=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )


def test_safe_set_mode_returns_surface():
    pygame = _Pygame()
    logger = _make_logger()

    screen = safe_set_mode((800, 600), pygame, logger, flags=pygame.RESIZABLE)

    assert screen is not None
    assert screen.get_size() == (800, 600)


def test_attempt_display_reinit_windowed_path():
    pygame = _Pygame()
    logger = _make_logger()
    gui = SimpleNamespace(
        fullscreen=False,
        _prev_window_size=(900, 700),
        _load_assets=lambda: None,
    )

    ok = attempt_display_reinit(gui, pygame, logger)

    assert ok is True
    assert gui.screen.get_size() == (900, 700)


def test_ensure_display_alive_healthy_surface():
    pygame = _Pygame()
    logger = _make_logger()
    toasts = []
    gui = SimpleNamespace(
        _show_toast=lambda msg, dur, level: toasts.append((msg, dur, level)),
        _display_recovery_attempts=3,
        fullscreen=False,
        _prev_window_size=(800, 600),
        _load_assets=lambda: None,
    )

    ok = ensure_display_alive(gui, pygame, logger)

    assert ok is True
    assert gui._display_recovery_attempts == 0
    assert toasts == []

