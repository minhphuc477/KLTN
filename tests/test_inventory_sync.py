import numpy as np
from simulation.validator import ZeldaLogicEnv, SEMANTIC_PALETTE
from gui_runner import ZeldaGUI


class DummyHUD:
    def __init__(self):
        self.last = None
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0

    def update_game_state(self, **kwargs):
        self.last = kwargs


def make_small_env():
    # Grid layout (3x4): S K L T
    S = SEMANTIC_PALETTE['START']
    K = SEMANTIC_PALETTE['KEY_SMALL']
    L = SEMANTIC_PALETTE['DOOR_LOCKED']
    T = SEMANTIC_PALETTE['TRIFORCE']
    F = SEMANTIC_PALETTE['FLOOR']

    grid = np.array([
        [F, F, F, F],
        [S, K, L, T],
        [F, F, F, F]
    ], dtype=np.int64)
    env = ZeldaLogicEnv(grid)
    env.start_pos = (1, 0)
    env.state.position = env.start_pos
    env.goal_pos = (1, 3)
    return env


def test_pickup_and_use_key_updates_hud():
    env = make_small_env()

    # Build a lightweight runner-like object
    runner = type('R', (), {})()
    runner.env = env
    runner.item_markers = {}
    runner.item_type_map = {}
    runner.collected_items = []
    runner.item_pickup_times = {}
    runner.keys_collected = 0
    runner.bombs_collected = 0
    runner.boss_keys_collected = 0
    runner.keys_used = 0
    runner.bombs_used = 0
    runner.boss_keys_used = 0
    runner.used_items = []
    runner.usage_effects = []

    # Attach helper methods from ZeldaGUI to the dummy object
    runner._sync_inventory_counters = ZeldaGUI._sync_inventory_counters.__get__(runner)
    runner._update_inventory_and_hud = ZeldaGUI._update_inventory_and_hud.__get__(runner)
    runner._apply_pickup_at = ZeldaGUI._apply_pickup_at.__get__(runner)
    runner._track_item_collection = ZeldaGUI._track_item_collection.__get__(runner)
    runner._track_item_usage = ZeldaGUI._track_item_usage.__get__(runner)
    runner._show_toast = lambda *a, **k: None
    runner.effects = None  # No effects in test environment
    runner.total_keys = 1
    runner.total_bombs = 0
    runner.total_boss_keys = 0

    # attach modern hud stub
    runner.modern_hud = DummyHUD()

    # Initially, no keys
    assert env.state.keys == 0

    # Simulate landing on key position
    key_pos = (1, 1)
    picked = runner._apply_pickup_at(key_pos)
    assert picked is True

    # After pickup, env should have key and HUD updated
    runner._update_inventory_and_hud()
    assert env.state.keys == 1
    assert runner.modern_hud.last is not None
    assert runner.modern_hud.last['keys'] == 1

    # Move to locked door and attempt to open (use key)
    # We simulate movement by modifying state (as env.step would do)
    old_state = env.state.copy()
    # open door at (1,2)
    door_pos = (1, 2)
    assert env.grid[door_pos] == SEMANTIC_PALETTE['DOOR_LOCKED']

    # Emulate using key to open door in the environment API
    can_move, new_state, reward, info = env._try_move(door_pos, env.grid[door_pos])
    assert can_move is True
    # Simulate env.step behavior: commit new_state into env.state and update grid
    env.state = new_state
    if env.grid[door_pos] == SEMANTIC_PALETTE['DOOR_LOCKED']:
        env.grid[door_pos] = SEMANTIC_PALETTE['DOOR_OPEN']
    # update gui runner tracking as in actual flow
    runner._track_item_usage(old_state, env.state)
    runner._track_item_collection(old_state, env.state)
    runner._update_inventory_and_hud()

    # Key should be consumed
    assert env.state.keys == 0 or getattr(new_state, 'keys', 0) == 0
    assert runner.keys_used >= 1
    assert runner.modern_hud.last['keys'] == getattr(env.state, 'keys', 0)
