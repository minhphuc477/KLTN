from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation import validator_rendering
from src.simulation.validator import ZeldaLogicEnv


class _Surface:
    def __init__(self, size=None):
        self.size = size
        self.fills = []
        self.blits = []

    def fill(self, color):
        self.fills.append(color)

    def blit(self, image, position):
        self.blits.append((image, position))


class _Font:
    def __init__(self):
        self.calls = []

    def render(self, text, antialias, color):
        rendered = (text, antialias, color)
        self.calls.append(rendered)
        return rendered


def _fake_pygame():
    calls = SimpleNamespace(
        init=0,
        quit=0,
        caption=None,
        flips=0,
        loaded=[],
        scaled=[],
        rects=[],
        font_args=None,
    )
    screen = _Surface()
    font = _Font()

    def init():
        calls.init += 1

    def quit_pygame():
        calls.quit += 1

    def set_mode(size):
        screen.size = size
        return screen

    def set_caption(caption):
        calls.caption = caption

    def sys_font(name, size, bold=False):
        calls.font_args = (name, size, bold)
        return font

    def load(path):
        calls.loaded.append(path)
        return _Surface()

    def scale(image, size):
        scaled = (image, size)
        calls.scaled.append(scaled)
        return scaled

    def draw_rect(surface, color, rect):
        calls.rects.append((surface, color, rect))

    def flip():
        calls.flips += 1

    pygame = SimpleNamespace(
        error=RuntimeError,
        init=init,
        quit=quit_pygame,
        Surface=_Surface,
        display=SimpleNamespace(
            set_mode=set_mode,
            set_caption=set_caption,
            flip=flip,
        ),
        font=SimpleNamespace(SysFont=sys_font),
        image=SimpleNamespace(load=load),
        transform=SimpleNamespace(scale=scale),
        draw=SimpleNamespace(rect=draw_rect),
    )
    return pygame, calls, screen, font


def _grid():
    return np.array(
        [
            [SEMANTIC_PALETTE["START"], SEMANTIC_PALETTE["FLOOR"]],
            [SEMANTIC_PALETTE["WALL"], SEMANTIC_PALETTE["TRIFORCE"]],
        ],
        dtype=np.int64,
    )


def test_renderer_delegates_initialize_load_draw_and_close(monkeypatch):
    pygame, calls, screen, font = _fake_pygame()
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setattr(
        validator_rendering.os.path,
        "exists",
        lambda path: path.endswith("floor.png") or path.endswith("link.png"),
    )

    env = ZeldaLogicEnv(_grid(), render_mode=True)

    assert calls.init == 1
    assert env.TILE_SIZE == 32
    assert screen.size == (64, 124)
    assert calls.caption == "ZAVE: Zelda Validation Environment"
    assert calls.font_args == ("Arial", 18, True)
    assert any(path.endswith("floor.png") for path in calls.loaded)
    assert any(path.endswith("link.png") for path in calls.loaded)
    assert env._images[SEMANTIC_PALETTE["WALL"]].fills == [(70, 70, 150)]

    env._link_img = None
    env.render()

    assert screen.fills == [(30, 30, 30)]
    assert len(screen.blits) == env.height * env.width + 2
    assert calls.rects[0][1:] == ((0, 255, 0), (4, 4, 24, 24))
    assert calls.rects[1][1:] == ((0, 0, 0), (0, 64, 64, 60))
    assert font.calls[-1][0] == "Playing..."
    assert calls.flips == 1

    env.close()
    assert calls.quit == 1


def test_renderer_disables_itself_when_pygame_is_unavailable(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pygame", None)

    env = ZeldaLogicEnv(_grid(), render_mode=True)

    assert env.render_mode is False
    assert env._screen is None
    assert env._font is None
    assert env._link_img is None
    assert env._images == {}
    assert "Pygame not available" in capsys.readouterr().out

    assert env.render() is None
    assert env.close() is None
