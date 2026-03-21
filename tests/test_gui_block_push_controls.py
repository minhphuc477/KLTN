import numpy as np

from src.gui.services.block_push_controls import (
    check_and_start_block_push,
    get_animating_block_positions,
    start_block_push_animation,
    update_block_push_animations,
)


class DummyLogger:
    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class DummyEffects:
    def __init__(self):
        self.items = []

    def add_effect(self, eff):
        self.items.append(eff)


class DummyPygame:
    class time:
        @staticmethod
        def get_ticks():
            return 1000

    SRCALPHA = 0


class DummyGui:
    def __init__(self):
        self.block_push_duration = 100
        self.block_push_animations = []
        self.effects = DummyEffects()
        self.env = type("E", (), {})()
        self.env.height = 3
        self.env.width = 4
        self.env.grid = np.array(
            [
                [1, 1, 1, 1],
                [1, 3, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int64,
        )
        self.started = []
        self.toasts = []

    def _start_block_push_animation(self, a, b):
        self.started.append((a, b))

    def _show_toast(self, msg, duration, toast_type):
        self.toasts.append((msg, duration, toast_type))


def _pop_effect(pos, color):
    return (pos, color)


def test_start_and_update_block_push_animation():
    gui = DummyGui()
    start_block_push_animation(gui, (1, 1), (1, 2), DummyPygame, DummyLogger())
    assert len(gui.block_push_animations) == 1

    # Force completion by old start time
    gui.block_push_animations[0]["start_time"] = 0
    update_block_push_animations(
        gui,
        DummyPygame,
        {"BLOCK": 3, "FLOOR": 1},
        _pop_effect,
        DummyLogger(),
    )
    assert gui.env.grid[1, 1] == 1
    assert gui.env.grid[1, 2] == 3


def test_check_and_start_block_push_and_positions_set():
    gui = DummyGui()
    ok = check_and_start_block_push(gui, (1, 0), (1, 1), {1}, {3})
    assert ok is True
    assert gui.started == [((1, 1), (1, 2))]
    assert gui.toasts[0][0] == "Block pushed!"

    gui.block_push_animations = [{"from_pos": (1, 1)}]
    assert get_animating_block_positions(gui) == {(1, 1)}

