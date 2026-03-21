from types import SimpleNamespace

from src.gui.controls.manual_step_controller import manual_step


class _Action:
    RIGHT = 3


class _Effects:
    def __init__(self):
        self.items = []

    def add_effect(self, eff):
        self.items.append(eff)


class _Renderer:
    def __init__(self):
        self.positions = []

    def set_agent_position(self, row, col, immediate=False):
        self.positions.append((row, col, immediate))


class _Env:
    def __init__(self):
        self.done = False
        self.won = False
        self.height = 4
        self.width = 4
        self.goal_pos = (0, 2)
        self.state = SimpleNamespace(
            position=(0, 0),
            keys=0,
            bomb_count=0,
            has_boss_key=False,
            has_bomb=False,
        )

    def step(self, _action):
        self.state.position = (0, 1)
        return self.state, 0.0, False, {"msg": "Moved"}


class _Time:
    @staticmethod
    def time():
        return 10.0


def _pop_effect(pos, color):
    return ("pop", pos, color)


def _flash_effect(pos, color, dur):
    return ("flash", pos, color, dur)


def test_manual_step_advances_position_and_updates_message():
    gui = SimpleNamespace(
        env=_Env(),
        step_count=0,
        keys_collected=0,
        total_keys=1,
        bombs_collected=0,
        boss_keys_collected=0,
        total_boss_keys=0,
        item_pickup_times={},
        message="",
        effects=_Effects(),
        modern_hud=None,
        renderer=_Renderer(),
        keys_used=0,
        bombs_used=0,
        boss_keys_used=0,
        _check_and_start_block_push=lambda *_args: None,
        _track_item_usage=lambda *_args: None,
    )

    manual_step(
        gui=gui,
        action=_Action.RIGHT,
        action_deltas={_Action.RIGHT: (0, 1)},
        pop_effect_cls=_pop_effect,
        flash_effect_cls=_flash_effect,
        time_module=_Time,
    )

    assert gui.step_count == 1
    assert gui.env.state.position == (0, 1)
    assert gui.message == "Moved"
    assert gui.renderer.positions[-1] == (0, 1, False)


def test_manual_step_done_win_sets_victory_message():
    env = _Env()

    def _step_win(_action):
        env.state.position = (0, 2)
        env.won = True
        return env.state, 1.0, True, {}

    env.step = _step_win

    gui = SimpleNamespace(
        env=env,
        step_count=0,
        keys_collected=0,
        total_keys=1,
        bombs_collected=0,
        boss_keys_collected=0,
        total_boss_keys=0,
        item_pickup_times={},
        message="",
        effects=_Effects(),
        modern_hud=None,
        renderer=_Renderer(),
        keys_used=0,
        bombs_used=0,
        boss_keys_used=0,
        _check_and_start_block_push=lambda *_args: None,
        _track_item_usage=lambda *_args: None,
    )

    manual_step(
        gui=gui,
        action=_Action.RIGHT,
        action_deltas={_Action.RIGHT: (0, 1)},
        pop_effect_cls=_pop_effect,
        flash_effect_cls=_flash_effect,
        time_module=_Time,
    )

    assert gui.message == "YOU WIN!"
    assert any(e[0] == "flash" for e in gui.effects.items)

