from types import SimpleNamespace

import numpy as np

from src.gui.map.loading import load_current_map


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class DummyEnv:
    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.start_pos = (0, 0)

    def _find_all_positions(self, tile_id):
        return list(zip(*np.where(self.grid == tile_id)))


class DummySanityChecker:
    def __init__(self, grid):
        self.grid = grid

    def check_all(self):
        return True, []


class DummyOs:
    environ = {}


SEMANTIC_PALETTE = {
    "KEY_SMALL": 10,
    "ITEM_MINOR": 11,
    "KEY_ITEM": 12,
    "KEY_BOSS": 13,
}


def _make_gui(auto_start_preview=False):
    gui = SimpleNamespace()
    gui.maps = [np.array([[0, 10], [11, 13]], dtype=np.int64)]
    gui.current_map_idx = 0
    gui.renderer = None
    gui.TILE_SIZE = 16
    gui.images = {}
    gui.auto_start_preview = auto_start_preview
    gui.preview_calls = 0
    gui._place_items_from_graph = lambda *args: None
    gui._auto_fit_zoom = lambda: None
    gui._center_view = lambda: None
    gui._start_preview_for_current_map = lambda: setattr(gui, "preview_calls", gui.preview_calls + 1)
    return gui


def test_load_current_map_does_not_auto_start_preview_by_default():
    gui = _make_gui(auto_start_preview=False)

    load_current_map(
        gui,
        os_module=DummyOs,
        logger=DummyLogger(),
        zelda_logic_env_cls=DummyEnv,
        sanity_checker_cls=DummySanityChecker,
        semantic_palette=SEMANTIC_PALETTE,
    )

    assert gui.message == "Map 1/1 - Press SPACE to solve"
    assert gui.preview_calls == 0


def test_load_current_map_can_auto_start_preview_when_enabled():
    gui = _make_gui(auto_start_preview=True)

    load_current_map(
        gui,
        os_module=DummyOs,
        logger=DummyLogger(),
        zelda_logic_env_cls=DummyEnv,
        sanity_checker_cls=DummySanityChecker,
        semantic_palette=SEMANTIC_PALETTE,
    )

    assert gui.preview_calls == 1
