from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace

from src.gui.services.auto_step_controller import auto_step, stop_auto


class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class _Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class _GameState:
    position: tuple
    keys: int
    bomb_count: int
    has_boss_key: bool
    opened_doors: set
    collected_items: set


class _Renderer:
    def __init__(self):
        self.calls = []

    def set_agent_position(self, row, col, immediate=False):
        self.calls.append((row, col, immediate))


class _Env:
    def __init__(self, start=(0, 0), goal=(1, 1)):
        self.state = SimpleNamespace(
            position=start,
            keys=0,
            bomb_count=0,
            has_boss_key=False,
            opened_doors=set(),
            collected_items=set(),
        )
        self.goal_pos = goal
        self.won = False
        self.done = False

    def step(self, _action):
        # Deterministic move right for this focused test.
        r, c = self.state.position
        self.state.position = (r, c + 1)
        return self.state, 0.0, False, {}


def test_stop_auto_clears_path_for_non_complete_reason():
    msgs = []
    gui = SimpleNamespace(
        auto_mode=True,
        _auto_stuck_retries=2,
        auto_path=[(0, 0), (0, 1)],
        auto_step_idx=1,
        _set_message=lambda m: msgs.append(m),
    )

    stop_auto(gui=gui, reason="manual", logger=_Logger())

    assert gui.auto_mode is False
    assert gui._auto_stuck_retries == 0
    assert gui.auto_path == []
    assert gui.auto_step_idx == 0
    assert msgs[-1] == "Auto-solve stopped: manual"


def test_auto_step_disables_auto_when_path_missing():
    errors = []
    gui = SimpleNamespace(
        auto_mode=True,
        auto_path=[],
        auto_step_idx=0,
        _show_error=lambda m: errors.append(m),
    )

    auto_step(
        gui=gui,
        logger=_Logger(),
        game_state_cls=_GameState,
        action_enum=_Action,
        ripple_effect_cls=object,
        flash_effect_cls=object,
        traceback_module=SimpleNamespace(print_exc=lambda: None),
    )

    assert gui.auto_mode is False
    assert errors[-1] == "No solution path available"


def test_auto_step_normal_move_advances_and_updates_renderer():
    track_calls = []
    renderer = _Renderer()
    env = _Env(start=(0, 0), goal=(2, 2))
    gui = SimpleNamespace(
        auto_mode=True,
        auto_path=[(0, 0), (0, 1)],
        auto_step_idx=0,
        feature_flags={"dstar_lite": False},
        dstar_active=False,
        dstar_solver=None,
        env=env,
        inventory_needs_refresh=False,
        _update_inventory_and_hud=lambda: None,
        _apply_pickup_at=lambda _p: None,
        _track_item_collection=lambda _old, _new: track_calls.append("collect"),
        _track_item_usage=lambda _old, _new: track_calls.append("use"),
        _sync_inventory_counters=lambda: None,
        _check_and_start_block_push=lambda *_args: None,
        _set_message=lambda _m: None,
        _show_error=lambda _m: None,
        status_message="",
        renderer=renderer,
        effects=None,
        modern_hud=None,
        step_count=0,
        keys_collected=0,
        bombs_collected=0,
        boss_keys_collected=0,
        keys_used=0,
        bombs_used=0,
        boss_keys_used=0,
        message="",
        _auto_stuck_retries=0,
        _stop_auto=lambda _reason=None: None,
    )

    auto_step(
        gui=gui,
        logger=_Logger(),
        game_state_cls=_GameState,
        action_enum=_Action,
        ripple_effect_cls=object,
        flash_effect_cls=object,
        traceback_module=SimpleNamespace(print_exc=lambda: None),
    )

    assert gui.auto_step_idx == 1
    assert gui.step_count == 1
    assert env.state.position == (0, 1)
    assert renderer.calls[-1] == (0, 1, False)
    assert track_calls == ["collect", "use"]
