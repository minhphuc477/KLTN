from types import SimpleNamespace

from src.gui.services.path_guaranteed_renderer import render_path_guaranteed


class _Surface:
    pass


class _Draw:
    def __init__(self):
        self.lines = []
        self.circles = []

    def line(self, surface, color, p1, p2, width):
        self.lines.append((surface, color, p1, p2, width))

    def circle(self, surface, color, center, radius, width=0):
        self.circles.append((surface, color, center, radius, width))


class _Pygame:
    def __init__(self):
        self.draw = _Draw()


class _Math:
    @staticmethod
    def sin(v):
        return 0.0


class _Time:
    def __init__(self):
        self._vals = [10.0, 10.1]

    def time(self):
        return self._vals.pop(0) if self._vals else 10.1


class _Logger:
    def __init__(self):
        self.debug_calls = []
        self.warn_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warn_calls.append((args, kwargs))


def test_render_path_guaranteed_draws_lines_and_markers():
    gui = SimpleNamespace(
        auto_path=[(0, 0), (0, 1), (1, 1)],
        _test_path=None,
        TILE_SIZE=16,
        view_offset_x=0,
        view_offset_y=0,
        path_item_positions={"keys": [(2, 2)]},
        collected_positions=set(),
        auto_step_idx=0,
    )
    surface = _Surface()
    pygame = _Pygame()
    logger = _Logger()

    render_path_guaranteed(
        gui=gui,
        surface=surface,
        pygame=pygame,
        math_module=_Math,
        time_module=_Time(),
        logger=logger,
    )

    assert len(pygame.draw.lines) >= 2
    assert len(pygame.draw.circles) >= 4


def test_render_path_guaranteed_ignores_invalid_path_point():
    gui = SimpleNamespace(
        auto_path=[(0, 0), "bad"],
        _test_path=None,
        TILE_SIZE=16,
        view_offset_x=0,
        view_offset_y=0,
        path_item_positions={},
        collected_positions=set(),
        auto_step_idx=0,
    )
    surface = _Surface()
    pygame = _Pygame()
    logger = _Logger()

    render_path_guaranteed(
        gui=gui,
        surface=surface,
        pygame=pygame,
        math_module=_Math,
        time_module=_Time(),
        logger=logger,
    )

    assert logger.warn_calls
    assert len(pygame.draw.lines) == 0

