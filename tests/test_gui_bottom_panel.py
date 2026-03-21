from types import SimpleNamespace

from src.gui.common.bottom_panel import (
    render_controls_section,
    render_inventory_section,
    render_message_section,
    render_metrics_section,
    render_progress_bar,
    render_status_section,
    render_unified_bottom_panel,
)


class _Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.width = w
        self.height = h


class _Rendered:
    def get_rect(self, **kwargs):
        return _Rect(kwargs.get("x", 0), kwargs.get("y", 0), kwargs.get("width", 10), kwargs.get("height", 10))


class _Font:
    def render(self, text, antialias, color):
        _ = (text, antialias, color)
        return _Rendered()


class _Surface:
    def __init__(self, size=(0, 0), flags=None):
        self.size = size
        self.flags = flags
        self.blits = []

    def fill(self, color):
        _ = color

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

    class draw:
        @staticmethod
        def rect(surface, color, rect, width=0, border_radius=0):
            _ = (surface, color, rect, width, border_radius)

        @staticmethod
        def line(surface, color, start, end, width=1):
            _ = (surface, color, start, end, width)


def _make_gui():
    screen = _Surface((800, 600))
    env_state = SimpleNamespace(keys=1, has_bomb=True, has_boss_key=False, position=(2, 3))
    env = SimpleNamespace(won=False, state=env_state, step_count=12)
    gui = SimpleNamespace(
        screen=screen,
        screen_w=800,
        screen_h=600,
        SIDEBAR_WIDTH=240,
        font=_Font(),
        small_font=_Font(),
        big_font=_Font(),
        env=env,
        message="hello world",
        step_count=7,
        speed_multiplier=1.0,
        TILE_SIZE=16,
        current_map_idx=0,
        map_names=["Dungeon A"],
        auto_mode=False,
        auto_path=[],
        auto_step_idx=0,
        status_message="Ready",
        keys_collected=1,
        total_keys=2,
        total_bombs=1,
        bombs_collected=1,
        total_boss_keys=1,
        boss_keys_collected=0,
        item_pickup_times={},
        collected_items=[],
    )
    gui._sync_inventory_counters = lambda: None
    return gui


def test_render_unified_bottom_panel_runs():
    gui = _make_gui()

    render_unified_bottom_panel(gui, _Pygame)

    assert len(gui.screen.blits) > 0


def test_render_progress_bar_runs_with_segments():
    surface = _Surface((200, 30))

    render_progress_bar(surface, 0, 0, 100, 10, 3, 5, (255, 0, 0), (10, 10, 10), _Pygame)

    assert isinstance(surface, _Surface)


def test_bottom_panel_sections_run():
    gui = _make_gui()
    logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    time_module = SimpleNamespace(time=lambda: 100.0)

    render_message_section(gui, 10, 10, 100, 40)
    render_inventory_section(gui, 10, 10, 120, 80, _Pygame, time_module, logger)
    render_metrics_section(gui, 10, 10, 120, 80)
    render_controls_section(gui, 10, 10, 160, 80)
    render_status_section(gui, 10, 10, 120, 80)

    assert len(gui.screen.blits) > 0

