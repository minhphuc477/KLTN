from types import SimpleNamespace

from src.gui.app.event_handlers.keyboard import (
    handle_global_keydown_shortcuts,
    handle_keydown_event,
)
from src.gui.app.main_loop_utils import run_continuous_movement_tick


class _Pygame:
    KEYDOWN = 1
    K_ESCAPE = 10
    K_F11 = 11
    K_h = 12
    K_F1 = 13
    K_TAB = 14
    K_F7 = 15
    K_F8 = 16
    K_m = 17
    K_g = 18
    K_BACKSPACE = 19
    K_RIGHTBRACKET = 20
    K_PERIOD = 21
    K_LEFTBRACKET = 22
    K_COMMA = 23
    K_SPACE = 24
    K_r = 25
    K_n = 26
    K_p = 27
    K_PLUS = 28
    K_EQUALS = 29
    K_MINUS = 30
    K_0 = 31
    K_f = 32
    K_c = 33
    K_l = 34
    K_UP = 35
    K_DOWN = 36
    K_LEFT = 37
    K_RIGHT = 38
    K_o = 39
    K_F12 = 40
    K_PAGEUP = 41
    K_PAGEDOWN = 42
    KMOD_CTRL = 0x01
    KMOD_SHIFT = 0x02

    class key:
        mods = 0

        @classmethod
        def get_mods(cls):
            return cls.mods

        @staticmethod
        def get_pressed():
            return {}


_Action = SimpleNamespace(
    UP="UP",
    DOWN="DOWN",
    LEFT="LEFT",
    RIGHT="RIGHT",
    UP_LEFT="UP_LEFT",
    UP_RIGHT="UP_RIGHT",
    DOWN_LEFT="DOWN_LEFT",
    DOWN_RIGHT="DOWN_RIGHT",
)


def _make_gui(diagonal=False):
    manual_steps = []
    centered = []
    return SimpleNamespace(
        auto_mode=False,
        keys_held={
            _Pygame.K_UP: False,
            _Pygame.K_DOWN: False,
            _Pygame.K_LEFT: False,
            _Pygame.K_RIGHT: False,
        },
        move_timer=0.0,
        move_delay=0.1,
        delta_time=0.2,
        feature_flags={"diagonal_movement": diagonal},
        _manual_step=manual_steps.append,
        _center_on_player=lambda: centered.append(True),
        manual_steps=manual_steps,
        centered=centered,
    )


def test_arrow_keydown_moves_immediately_for_manual_play():
    gui = _make_gui()
    event = SimpleNamespace(key=_Pygame.K_RIGHT, repeat=False)

    running = handle_keydown_event(gui, event, _Pygame, SimpleNamespace(path=SimpleNamespace()), None, object, _Action, True)

    assert running is True
    assert gui.manual_steps == [_Action.RIGHT]
    assert len(gui.centered) == 1
    assert gui.keys_held[_Pygame.K_RIGHT] is True


def test_arrow_keydown_repeat_does_not_double_step():
    gui = _make_gui()
    event = SimpleNamespace(key=_Pygame.K_RIGHT, repeat=True)

    handle_keydown_event(gui, event, _Pygame, SimpleNamespace(path=SimpleNamespace()), None, object, _Action, True)

    assert gui.manual_steps == []
    assert gui.keys_held[_Pygame.K_RIGHT] is True


def test_manual_diagonal_keydown_respects_feature_flag():
    gui = _make_gui(diagonal=False)
    gui.keys_held[_Pygame.K_UP] = True
    event = SimpleNamespace(key=_Pygame.K_RIGHT, repeat=False)

    handle_keydown_event(gui, event, _Pygame, SimpleNamespace(path=SimpleNamespace()), None, object, _Action, True)

    assert gui.manual_steps == [_Action.RIGHT]

    gui = _make_gui(diagonal=True)
    gui.keys_held[_Pygame.K_UP] = True
    handle_keydown_event(gui, event, _Pygame, SimpleNamespace(path=SimpleNamespace()), None, object, _Action, True)

    assert gui.manual_steps == [_Action.UP_RIGHT]


def test_hold_to_move_respects_diagonal_flag_and_recenters():
    gui = _make_gui(diagonal=False)
    gui.keys_held[_Pygame.K_UP] = True
    gui.keys_held[_Pygame.K_RIGHT] = True

    run_continuous_movement_tick(gui, _Pygame, _Action)

    assert gui.manual_steps == [_Action.UP]
    assert len(gui.centered) == 1

    gui = _make_gui(diagonal=True)
    gui.keys_held[_Pygame.K_UP] = True
    gui.keys_held[_Pygame.K_RIGHT] = True
    run_continuous_movement_tick(gui, _Pygame, _Action)

    assert gui.manual_steps == [_Action.UP_RIGHT]
    assert len(gui.centered) == 1


def test_shift_f12_toggles_panel_debug_instead_of_overlay():
    messages = []
    gui = SimpleNamespace(
        debug_control_panel=False,
        debug_overlay_enabled=False,
        _set_message=lambda message: messages.append(message),
    )
    _Pygame.key.mods = _Pygame.KMOD_SHIFT
    try:
        consumed = handle_global_keydown_shortcuts(
            gui,
            SimpleNamespace(type=_Pygame.KEYDOWN, key=_Pygame.K_F12),
            _Pygame,
            SimpleNamespace(time=lambda: 0.0),
            SimpleNamespace(debug=lambda *_args, **_kwargs: None),
            object,
        )
    finally:
        _Pygame.key.mods = 0

    assert consumed is True
    assert gui.debug_control_panel is True
    assert gui.debug_overlay_enabled is False
    assert messages == ["Control panel debug ON (Shift+F12)"]
