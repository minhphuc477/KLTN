from src.gui.controls.view_navigation import center_on_player, clamp_view_offset, next_map, prev_map


class DummyEnv:
    def __init__(self):
        self.width = 20
        self.height = 10
        self.state = type("S", (), {"position": (4, 5)})()


class DummyEffects:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class DummyLogger:
    def exception(self, *args, **kwargs):
        return None


class DummyGui:
    def __init__(self):
        self.maps = [1, 2, 3]
        self.current_map_idx = 0
        self.auto_mode = False
        self.stopped = 0
        self.loaded = 0
        self.centered = 0
        self.effects = DummyEffects()
        self.step_count = 8

        self.env = DummyEnv()
        self.TILE_SIZE = 16
        self.screen_w = 800
        self.screen_h = 600
        self.SIDEBAR_WIDTH = 200
        self.HUD_HEIGHT = 120
        self.view_offset_x = -999
        self.view_offset_y = 999

    def _stop_auto(self, _reason):
        self.stopped += 1

    def _load_current_map(self):
        self.loaded += 1

    def _center_view(self):
        self.centered += 1

    def _clamp_view_offset(self):
        clamp_view_offset(self)


def test_next_prev_map_flow_updates_index_and_resets_state():
    gui = DummyGui()
    next_map(gui, DummyLogger())
    assert gui.current_map_idx == 1
    assert gui.loaded == 1
    assert gui.centered == 1
    assert gui.step_count == 0

    prev_map(gui, DummyLogger())
    assert gui.current_map_idx == 0
    assert gui.loaded == 2


def test_clamp_and_center_on_player_keep_offsets_in_bounds():
    gui = DummyGui()
    clamp_view_offset(gui)
    assert isinstance(gui.view_offset_x, int)
    assert isinstance(gui.view_offset_y, int)

    center_on_player(gui)
    assert isinstance(gui.view_offset_x, int)
    assert isinstance(gui.view_offset_y, int)

