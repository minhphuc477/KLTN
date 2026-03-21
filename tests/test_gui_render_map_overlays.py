from types import SimpleNamespace

from src.gui.rendering.render_map_overlays import (
    log_draw_ranges,
    render_empty_range_warning,
    render_jps_overlay,
    render_map_elites_overlay,
)


class _Text:
    def get_width(self):
        return 40

    def get_height(self):
        return 10


class _Font:
    def render(self, _text, _aa, _color):
        return _Text()


class _Screen:
    def __init__(self):
        self.blits = []

    def blit(self, obj, pos):
        self.blits.append((obj, pos))


class _MapSurface:
    def __init__(self):
        self.blits = []

    def blit(self, obj, pos):
        self.blits.append((obj, pos))


class _Draw:
    def __init__(self):
        self.lines = []
        self.rects = []

    def line(self, surface, color, p1, p2, width):
        self.lines.append((surface, color, p1, p2, width))

    def rect(self, surface, color, rect):
        self.rects.append((surface, color, rect))


class _MiniSurf:
    def fill(self, _color):
        return None


class _Pygame:
    SRCALPHA = 1

    def __init__(self):
        self.draw = _Draw()
        self.font = SimpleNamespace(SysFont=lambda *_args, **_kwargs: _Font())

    def Surface(self, _size, _flags=None):
        return _MiniSurf()


class _Time:
    @staticmethod
    def time():
        return 10.0


class _Logger:
    def __init__(self):
        self.debug_calls = []
        self.exc_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exc_calls.append((args, kwargs))


def test_log_draw_ranges_emits_debug_once():
    gui = SimpleNamespace(
        _last_render_log_time=0.0,
        env=SimpleNamespace(grid=[[1, 1], [1, 1]]),
        images={1: object()},
    )
    logger = _Logger()

    log_draw_ranges(
        gui=gui,
        start_r=0,
        end_r=1,
        start_c=0,
        end_c=1,
        h=2,
        w=2,
        time_module=_Time,
        logger=logger,
    )

    assert logger.debug_calls


def test_render_jps_overlay_draws_segments_and_jumps():
    map_surface = _MapSurface()
    pygame = _Pygame()
    gui = SimpleNamespace(
        feature_flags={"show_jps_overlay": True},
        last_jps_trace={"segments": [((0, 0), (0, 1))], "jumps": [(0, 1)]},
        TILE_SIZE=16,
        view_offset_x=0,
        view_offset_y=0,
    )

    render_jps_overlay(
        gui=gui,
        map_surface=map_surface,
        start_r=0,
        end_r=2,
        start_c=0,
        end_c=2,
        pygame=pygame,
    )

    assert len(pygame.draw.lines) == 1
    assert len(map_surface.blits) == 1


def test_render_map_elites_overlay_blits_surface():
    map_surface = _MapSurface()
    pygame = _Pygame()

    class _Occ:
        shape = (2, 2)

        def __getitem__(self, key):
            r, c = key
            return 1 if (r + c) % 2 == 0 else 0

    gui = SimpleNamespace(
        feature_flags={"show_map_elites": True},
        map_elites_result=SimpleNamespace(occupancy_grid=lambda: _Occ()),
        screen_w=1000,
        SIDEBAR_WIDTH=220,
        view_offset_x=0,
        view_offset_y=0,
    )

    render_map_elites_overlay(gui=gui, map_surface=map_surface, pygame=pygame)

    assert len(map_surface.blits) == 1


def test_render_empty_range_warning_blits_when_invalid_range():
    gui = SimpleNamespace(
        screen=_Screen(),
        screen_w=800,
        screen_h=600,
        big_font=_Font(),
    )
    pygame = _Pygame()

    render_empty_range_warning(
        gui=gui,
        start_r=2,
        end_r=1,
        start_c=0,
        end_c=0,
        pygame=pygame,
    )

    assert gui.screen.blits

