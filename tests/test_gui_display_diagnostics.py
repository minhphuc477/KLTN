from types import SimpleNamespace

from src.gui.overlay.display_diagnostics import handle_watchdog_screenshot, report_ui_state


class _Display:
    def __init__(self, surface=None):
        self._surface = surface

    def get_surface(self):
        return self._surface


class _Image:
    def __init__(self):
        self.saved = []

    def save(self, surf, path):
        self.saved.append((surf, path))


class _Pygame:
    def __init__(self, surface=None):
        self.display = _Display(surface)
        self.image = _Image()


class _Proc:
    def is_alive(self):
        return True


def test_handle_watchdog_screenshot_saves_and_clears_request():
    gui = SimpleNamespace(_watchdog_request_screenshot="tmp/test.png")
    toasts = []
    gui._show_toast = lambda msg, dur, typ: toasts.append((msg, dur, typ))
    pygame = _Pygame(surface=object())
    logger = SimpleNamespace(warning=lambda *a, **k: None, exception=lambda *a, **k: None)
    os_module = SimpleNamespace(path=SimpleNamespace(basename=lambda p: "test.png"))

    ok = handle_watchdog_screenshot(gui, pygame, logger, os_module)

    assert ok is True
    assert gui._watchdog_request_screenshot is None
    assert pygame.image.saved and pygame.image.saved[0][1] == "tmp/test.png"
    assert toasts and "Watchdog screenshot" in toasts[0][0]


def test_report_ui_state_returns_expected_fields():
    gui = SimpleNamespace(
        fullscreen=False,
        screen_w=800,
        screen_h=600,
        preview_overlay_visible=False,
        preview_modal_enabled=False,
        control_panel_enabled=True,
        control_panel_rect=(1, 2, 3, 4),
        control_panel_collapsed=False,
        solver_running=True,
        solver_proc=_Proc(),
        solver_comparison_thread=None,
        debug_click_log=[1, 2, 3],
    )
    logger = SimpleNamespace(exception=lambda *a, **k: None)

    state = report_ui_state(gui, logger)

    assert state["screen_w"] == 800
    assert state["control_panel_enabled"] is True
    assert state["solver_proc_alive"] is True
    assert state["debug_click_log_len"] == 3

