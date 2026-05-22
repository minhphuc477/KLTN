from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.gui.services.auto_step_controller import auto_step, stop_auto
from src.simulation.validator import ZeldaLogicEnv


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
    def __init__(self, start=(0, 0), goal=(1, 1), grid=None):
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
        self.grid = grid if grid is not None else np.full((3, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int32)

    def step(self, _action):
        # Deterministic move right for this focused test.
        r, c = self.state.position
        self.state.position = (r, c + 1)
        return self.state, 0.0, False, {}


class _DoorDesyncEnv(_Env):
    def __init__(self):
        grid = np.array([[SEMANTIC_PALETTE["FLOOR"], SEMANTIC_PALETTE["DOOR_BOMB"]]], dtype=np.int32)
        super().__init__(start=(0, 0), goal=(0, 1), grid=grid)
        self.state.bomb_count = 1

    def step(self, _action):
        # Simulate a replay/backend desync where the bomb is consumed and the
        # door is opened, but position remains on the source tile for one tick.
        self.state.bomb_count = 0
        self.state.opened_doors.add((0, 1))
        self.grid[0, 1] = SEMANTIC_PALETTE["DOOR_OPEN"]
        return self.state, 9.9, False, {"msg": "Bombed wall"}


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


def test_auto_step_expands_non_adjacent_transition_into_grid_steps():
    messages = []
    renderer = _Renderer()
    grid = np.full((1, 4), SEMANTIC_PALETTE["FLOOR"], dtype=np.int32)
    env = _Env(start=(0, 0), goal=(0, 3), grid=grid)
    gui = SimpleNamespace(
        auto_mode=True,
        auto_path=[(0, 0), (0, 3)],
        auto_step_idx=0,
        feature_flags={"dstar_lite": False},
        dstar_active=False,
        dstar_solver=None,
        env=env,
        inventory_needs_refresh=False,
        _update_inventory_and_hud=lambda: None,
        _track_item_collection=lambda _old, _new: None,
        _track_item_usage=lambda _old, _new: None,
        _sync_inventory_counters=lambda: None,
        _check_and_start_block_push=lambda *_args: None,
        _set_message=lambda m: messages.append(m),
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

    assert gui.auto_path == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert gui.auto_step_idx == 1
    assert env.state.position == (0, 1)
    assert renderer.calls[-1] == (0, 1, False)
    assert messages[0] == "Expanded solver shortcut into grid steps"


def test_auto_step_stops_non_adjacent_transition_without_grid_route():
    errors = []
    stopped = []
    grid = np.array(
        [[SEMANTIC_PALETTE["FLOOR"], SEMANTIC_PALETTE["WALL"], SEMANTIC_PALETTE["FLOOR"]]],
        dtype=np.int32,
    )
    env = _Env(start=(0, 0), goal=(0, 2), grid=grid)
    gui = SimpleNamespace(
        auto_mode=True,
        auto_path=[(0, 0), (0, 2)],
        auto_step_idx=0,
        feature_flags={"dstar_lite": False},
        dstar_active=False,
        dstar_solver=None,
        env=env,
        inventory_needs_refresh=False,
        _update_inventory_and_hud=lambda: None,
        _set_message=lambda _m: None,
        _show_error=lambda m: errors.append(m),
        status_message="",
        renderer=None,
        effects=None,
        modern_hud=None,
        step_count=0,
        message="",
        _auto_stuck_retries=0,
        _stop_auto=lambda reason=None: stopped.append(reason),
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

    assert "non-adjacent transition without a grid route" in errors[-1]
    assert stopped == ["non-adjacent replay transition"]
    assert env.state.position == (0, 0)


def test_auto_step_does_not_expand_shortcut_through_locked_boss_door_without_key():
    errors = []
    stopped = []
    grid = np.array(
        [[SEMANTIC_PALETTE["START"], SEMANTIC_PALETTE["DOOR_BOSS"], SEMANTIC_PALETTE["TRIFORCE"]]],
        dtype=np.int32,
    )
    env = ZeldaLogicEnv(grid)
    gui = SimpleNamespace(
        auto_mode=True,
        auto_path=[(0, 0), (0, 2)],
        auto_step_idx=0,
        feature_flags={"dstar_lite": False},
        dstar_active=False,
        dstar_solver=None,
        env=env,
        inventory_needs_refresh=False,
        _update_inventory_and_hud=lambda: None,
        _set_message=lambda _m: None,
        _show_error=lambda m: errors.append(m),
        status_message="",
        renderer=None,
        effects=None,
        modern_hud=None,
        step_count=0,
        message="",
        _auto_stuck_retries=0,
        _stop_auto=lambda reason=None: stopped.append(reason),
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

    assert "non-adjacent transition without a grid route" in errors[-1]
    assert stopped == ["non-adjacent replay transition"]
    assert env.state.position == (0, 0)


def test_auto_step_recovers_opened_door_position_desync():
    usage_calls = []
    renderer = _Renderer()
    env = _DoorDesyncEnv()
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
        _track_item_collection=lambda _old, _new: None,
        _track_item_usage=lambda _old, _new: usage_calls.append((_old.bomb_count, _new.bomb_count, _new.position)),
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

    assert env.state.position == (0, 1)
    assert env.grid[0, 1] == SEMANTIC_PALETTE["DOOR_OPEN"]
    assert gui.auto_step_idx == 1
    assert gui._auto_stuck_retries == 0
    assert usage_calls == [(1, 0, (0, 1))]
    assert renderer.calls[-1] == (0, 1, False)
